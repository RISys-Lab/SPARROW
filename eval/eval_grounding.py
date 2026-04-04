import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import skimage
import torch
import torch.nn.functional as F
from tqdm import tqdm

from chat import (
    IMAGE_TOKEN_INDEX,
    TSF_TOKEN_INDEX,
    initialize_model_videogptplus,
    preprocess_tsf_crop,
    preprocess_tsf_from_bbox_txt,
    preprocess_vision,
)
from util.grounding_utils.box_ops import masks_to_boxes, np_box_iou
from util.grounding_utils.image_transforms import make_video_transforms
from util.hcstvg_dataset import VideoModulatedSTGrounding_HCSTVGv2
from util.vidstg_dataset import VideoModulatedSTGrounding

iou_thresholds = [0.3, 0.5]


def summarize_metrics(results, tmp_loc):
    if not results:
        print("No metrics to summarize.")
        return {"vid_metrics": {}, "summary": {}}

    categories = sorted(set(x["qtype"] for x in results.values()))
    metrics = {}
    counter = {}
    for category in categories:
        metrics[category] = {"gt_viou": 0.0}
        if tmp_loc:
            metrics[category].update({"tiou": 0.0, "viou": 0.0})
        for thresh in iou_thresholds:
            if tmp_loc:
                metrics[category][f"viou@{thresh}"] = 0.0
            metrics[category][f"gt_viou@{thresh}"] = 0.0
        counter[category] = 0

    for x in results.values():
        qtype = x["qtype"]
        if tmp_loc:
            metrics[qtype]["tiou"] += x["tiou"]
            metrics[qtype]["viou"] += x["viou"]
        metrics[qtype]["gt_viou"] += x["gt_viou"]
        for thresh in iou_thresholds:
            if tmp_loc:
                metrics[qtype][f"viou@{thresh}"] += x[f"viou@{thresh}"]
            metrics[qtype][f"gt_viou@{thresh}"] += x[f"gt_viou@{thresh}"]
        counter[qtype] += 1

    for category in categories:
        denom = max(counter[category], 1)
        for key in metrics[category]:
            metrics[category][key] = metrics[category][key] / denom
            print(f"{category} {key}: {metrics[category][key]:.4f}")

    return {"vid_metrics": results, "summary": metrics}


def _calc_tiou(gt_sted, pred_sted, frame_ids):
    max_start = max(gt_sted[0], pred_sted[0])
    min_end = min(gt_sted[1], pred_sted[1])
    min_start = min(gt_sted[0], pred_sted[0])
    max_end = max(gt_sted[1], pred_sted[1])
    if min_end <= max_start:
        tiou = 0
    else:
        intersection = min_end - max_start
        gt_span = gt_sted[1] - gt_sted[0]
        pred_span = pred_sted[1] - pred_sted[0]
        union = gt_span + pred_span - intersection
        tiou = intersection / union

    union_predgt = [frame_id for frame_id in frame_ids if min_start <= frame_id < max_end]
    inter_predgt = set(frame_id for frame_id in frame_ids if max_start <= frame_id < min_end)

    return tiou, union_predgt, inter_predgt


def remove_small_blobs(binary_mask: np.ndarray, min_size: int = 0):
    if min_size > 0:
        dtype = binary_mask.dtype
        binary_mask = skimage.morphology.remove_small_objects(binary_mask.astype(bool), min_size=min_size)
        binary_mask = binary_mask.astype(dtype)
    return binary_mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run spatio-temporal grounding evaluation inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--llava_version_or_path", type=str, default="hf_model", help="Path to model directory.")
    parser.add_argument("--vis_save_path", type=str, default="vis_output/eval_grounding", help="Output root for predicted masks and metrics.")
    parser.add_argument("--precision", type=str, default="fp16", choices=["bf16", "fp16", "fp32"], help="Inference precision.")
    parser.add_argument("--local_rank", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--load_in_8bit", action="store_true", default=False, help="Load LLM in 8-bit.")
    parser.add_argument("--load_in_4bit", action="store_true", default=False, help="Load LLM in 4-bit.")
    parser.add_argument("--use_sam2_video_branch", action="store_true", default=False, help="Use SAM2 video branch at inference.")
    parser.add_argument("--base_model_type", type=str, default="vgpt|phi3", choices=["vgpt|phi3"], help="Model family.")
    parser.add_argument("--use_tsf_token", action="store_true", default=False, help="Enable TSF token pathway.")
    parser.add_argument("--tsf_k", type=int, default=4, help="Number of TSF representatives after k-means.")
    parser.add_argument("--tsf_kmeans_iters", type=int, default=8, help="TSF k-means iterations.")
    parser.add_argument("--tsf_crop_path", type=str, default="", help="Static TSF crop image path used for all videos.")
    parser.add_argument(
        "--tsf_boxes_txt",
        type=str,
        default="",
        help="TSF bbox txt path, directory, or template containing {video} (e.g., /path/{video}.txt).",
    )

    parser.set_defaults(
        use_detection_guidance=True,
        use_roi_helper=True,
        use_box_prior_filter=True,
    )
    parser.add_argument("--use_detection_guidance", dest="use_detection_guidance", action="store_true", help="Enable detection-guided prompting.")
    parser.add_argument("--disable_detection_guidance", dest="use_detection_guidance", action="store_false", help="Disable detection-guided prompting.")
    parser.add_argument("--ddetr_model_path", type=str, default="checkpoints_hf/ddetr_sam2", help="Path to DDETR checkpoint.")
    parser.add_argument("--proposal_topk", type=int, default=100, help="Number of DDETR proposals to keep.")
    parser.add_argument("--proposal_score_power", type=float, nargs=2, default=(0.3, 0.7), help="Score fusion exponents.")
    parser.add_argument("--proposal_score_threshold", type=float, default=0.4, help="Proposal selection threshold.")
    parser.add_argument("--use_roi_helper", dest="use_roi_helper", action="store_true", help="Enable ROI helper head.")
    parser.add_argument("--disable_roi_helper", dest="use_roi_helper", action="store_false", help="Disable ROI helper head.")
    parser.add_argument("--roi_helper_hidden_dim", type=int, default=256, help="ROI helper hidden size.")
    parser.add_argument("--roi_align_output_size", type=int, default=7, help="ROIAlign output size.")
    parser.add_argument("--roi_fusion_init", type=float, default=0.1, help="Initial ROI fusion weight.")
    parser.add_argument("--use_box_prior_filter", dest="use_box_prior_filter", action="store_true", help="Enable prior-based proposal filtering.")
    parser.add_argument("--disable_box_prior_filter", dest="use_box_prior_filter", action="store_false", help="Disable prior-based proposal filtering.")
    parser.add_argument("--box_prior_min_score", type=float, default=0.35, help="Minimum prior score for selected proposals.")
    parser.add_argument("--box_prior_max_keep", type=int, default=1, help="Max boxes per token after prior filtering.")

    parser.add_argument(
        "--video_dataset_dir",
        type=str,
        default="/media/ansari/BFEE-795F/VideoGLaMM/video_dataset/",
        help="Dataset root.",
    )
    parser.add_argument("--dataset_name", default="vidstg", type=str, choices=["vidstg", "hcstvg"], help="Grounding benchmark.")
    parser.add_argument("--tmp_loc", action="store_true", default=False, help="Evaluate temporal localization as well as spatial localization.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new text tokens during generation.")
    parser.add_argument("--print_text_output", action="store_true", default=False, help="Print decoded model text for each sample.")
    return parser.parse_args()


def initialize_runtime(args):
    base_type, llm_type = args.base_model_type.split("|")
    if base_type != "vgpt":
        raise ValueError(f"Unsupported base model family: {args.base_model_type}")
    return initialize_model_videogptplus(
        model_base=args.llava_version_or_path,
        precision=args.precision,
        local_rank=args.local_rank,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        use_sam2_video_branch=args.use_sam2_video_branch,
        base_llm_type=llm_type,
        use_detection_guidance=args.use_detection_guidance,
        ddetr_model_path=args.ddetr_model_path,
        proposal_topk=args.proposal_topk,
        proposal_score_power=args.proposal_score_power,
        proposal_score_threshold=args.proposal_score_threshold,
        use_roi_helper=args.use_roi_helper,
        roi_helper_hidden_dim=args.roi_helper_hidden_dim,
        roi_align_output_size=args.roi_align_output_size,
        roi_fusion_init=args.roi_fusion_init,
        use_box_prior_filter=args.use_box_prior_filter,
        box_prior_min_score=args.box_prior_min_score,
        box_prior_max_keep=args.box_prior_max_keep,
        use_tsf_token=args.use_tsf_token,
        tsf_k=args.tsf_k,
        tsf_kmeans_iters=args.tsf_kmeans_iters,
    )


def _resolve_tsf_boxes_path(tsf_boxes_txt_arg, video_name):
    if not tsf_boxes_txt_arg:
        return ""
    if "{video}" in tsf_boxes_txt_arg:
        return tsf_boxes_txt_arg.format(video=video_name)
    if os.path.isdir(tsf_boxes_txt_arg):
        return os.path.join(tsf_boxes_txt_arg, f"{video_name}.txt")
    return tsf_boxes_txt_arg


def _first_mask_for_frame(frame_preds, target_hw):
    h, w = int(target_hw[0]), int(target_hw[1])
    if not isinstance(frame_preds, dict) or len(frame_preds) == 0:
        return np.zeros((h, w), dtype=np.bool_)

    first_obj_id = sorted(frame_preds.keys())[0]
    mask_t = torch.as_tensor(frame_preds[first_obj_id], dtype=torch.float32)
    if mask_t.ndim == 3:
        mask_t = mask_t[0]
    if mask_t.ndim != 2:
        return np.zeros((h, w), dtype=np.bool_)
    if tuple(mask_t.shape[-2:]) != (h, w):
        mask_t = F.interpolate(mask_t.unsqueeze(0).unsqueeze(0), size=(h, w), mode="nearest")[0, 0]
    return (mask_t > 0).detach().cpu().numpy().astype(np.bool_)


def _load_eval_dataset(args):
    base_video_dataset_dir = args.video_dataset_dir
    tmp_loc = args.tmp_loc
    print(f"tmp_loc: {tmp_loc}")

    if args.dataset_name == "vidstg":
        vidstg_vid_dir = os.path.join(base_video_dataset_dir, "vidstg/video")
        vidstg_ann_dir = os.path.join(base_video_dataset_dir, "processed/vidstg/vidstg_annotations")
        vidstg_ann_file = os.path.join(vidstg_ann_dir, "test.json")

        image_size = 224
        sample_fps = 1
        max_num_frames = 40

        return VideoModulatedSTGrounding(
            vidstg_vid_dir,
            vidstg_ann_file,
            transforms=make_video_transforms("test", cautious=True, resolution=image_size, normalize=False),
            is_train=False,
            video_max_len=max_num_frames,
            video_max_len_train=max_num_frames,
            fps=sample_fps,
            tmp_crop=False,
            tmp_loc=tmp_loc,
        )

    if args.dataset_name == "hcstvg":
        vid_folder = os.path.join(base_video_dataset_dir, "hcstvg", "Video")
        processed_ann_dir = os.path.join(base_video_dataset_dir, "processed/hcstvg/hcstvg_annotations")
        ann_file = os.path.join(processed_ann_dir, "val_v2_proc.json")

        image_size = 224
        sample_fps = 1
        max_num_frames = 40

        return VideoModulatedSTGrounding_HCSTVGv2(
            vid_folder,
            ann_file,
            transforms=make_video_transforms("val", cautious=True, resolution=image_size, normalize=False),
            is_train=False,
            video_max_len=max_num_frames,
            video_max_len_train=max_num_frames,
            fps=sample_fps,
            tmp_crop=False,
            tmp_loc=tmp_loc,
        )

    raise ValueError(f"Invalid dataset name: {args.dataset_name}")


def main():
    args = parse_args()

    model, tokenizer, enc_preprocessor, conv_generator, sam_preprocessor = initialize_runtime(args)
    static_tsf_images = None
    if args.use_tsf_token and args.tsf_crop_path:
        static_tsf_images = preprocess_tsf_crop(
            tsf_crop_path=args.tsf_crop_path,
            enc_preprocessor=enc_preprocessor,
            precision=args.precision,
        )

    eval_dataset = _load_eval_dataset(args)

    autocast_enabled = args.precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    vid_metrics = {}

    for idx in tqdm(range(len(eval_dataset)), desc="Processing videos"):
        try:
            save_dir_for_current_video = os.path.join(args.vis_save_path, args.dataset_name, f"{idx:06d}")
            os.makedirs(save_dir_for_current_video, exist_ok=True)
            metrics_file = os.path.join(save_dir_for_current_video, "metrics.pt")
            if os.path.exists(metrics_file):
                print(f"Skipping {idx} as it already exists.")
                vid_metrics[idx] = torch.load(metrics_file)
                continue

            vid_path, images, targets, tmp_target = eval_dataset[idx]
            time_s, time_e = tmp_target["inter_idx"]
            caption, qtype = tmp_target["caption"], tmp_target["qtype"]
            gt_boxes_per_video = targets

            np_images = [(img * 255).numpy().astype("uint8") for img in images]
            np_images_batch = [np_images]

            if args.dataset_name == "hcstvg":
                hcstvg_qa_dir = os.path.join(args.video_dataset_dir, "hcstvg", "qa")
                qa_file = os.path.join(hcstvg_qa_dir, f"{idx}.json")
                with open(qa_file, "r") as f:
                    res_dict = json.load(f)
                question = res_dict["Q"]
                answer = res_dict["A"]
                assert question != "" and answer != ""
                caption = question
                qtype = "interrogative"

            if qtype == "interrogative":
                prompt_text = f"{caption} Please respond with segmentation masks."
            else:
                prompt_text = f"Can you segment {caption} in this video?"

            enc_image, enc_context_image, image_sam, original_size_list, resize_list = preprocess_vision(
                np_images=np_images_batch,
                type="video",
                enc_preprocessor=enc_preprocessor,
                sam_preprocessor=sam_preprocessor,
                conv_generator=conv_generator,
                precision=args.precision,
            )
            input_ids = conv_generator.apply_for_chat(prompt_text, type="video", tokenizer=tokenizer)

            video_key = Path(str(vid_path)).stem
            if video_key == "":
                video_key = f"{idx:06d}"
            tsf_images = static_tsf_images
            if args.use_tsf_token and tsf_images is None and args.tsf_boxes_txt:
                tsf_boxes_path = _resolve_tsf_boxes_path(args.tsf_boxes_txt, video_key)
                if tsf_boxes_path and os.path.isfile(tsf_boxes_path):
                    tsf_images = preprocess_tsf_from_bbox_txt(
                        tsf_boxes_txt=tsf_boxes_path,
                        np_images=np_images_batch,
                        input_type="video",
                        conv_generator=conv_generator,
                        enc_preprocessor=enc_preprocessor,
                        precision=args.precision,
                    )

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                output_ids_batch, video_segments_batch = model.inference(
                    images=enc_image,
                    context_images=enc_context_image,
                    images_for_sam=image_sam,
                    tsf_images=tsf_images,
                    input_ids=input_ids,
                    resize_list=resize_list,
                    original_size_list=original_size_list,
                    max_new_tokens=args.max_new_tokens,
                    use_sam2_video_branch=args.use_sam2_video_branch,
                )

            assert len(output_ids_batch) == 1 and len(video_segments_batch) == 1, "Batch size must be 1"
            output_ids = output_ids_batch[0]
            output_ids = output_ids[output_ids != IMAGE_TOKEN_INDEX]
            output_ids = output_ids[output_ids != TSF_TOKEN_INDEX]
            text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
            text_output = text_output.replace("\n", "").replace("  ", " ")
            if args.print_text_output:
                print(f"text_output ({idx}): {text_output}")

            video_segments = video_segments_batch[0]
            predictions = {}
            for t in range(len(np_images)):
                pred_mask_i = _first_mask_for_frame(
                    frame_preds=video_segments.get(t, {}) if isinstance(video_segments, dict) else {},
                    target_hw=np_images[t].shape[:2],
                )
                pred_mask_i = remove_small_blobs(pred_mask_i, min_size=20)
                pred_boxes = masks_to_boxes(torch.tensor(np.expand_dims(pred_mask_i, axis=0)))
                predictions[t] = {"boxes": pred_boxes}

                save_path = os.path.join(save_dir_for_current_video, f"mask_{t}_0.jpg")
                cv2.imwrite(save_path, pred_mask_i.astype(np.uint8) * 100)

                save_path = os.path.join(save_dir_for_current_video, f"masked_img_{t}_0.jpg")
                save_img = np_images[t].copy()
                save_img[pred_mask_i] = (
                    np_images[t] * 0.5
                    + pred_mask_i[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
                )[pred_mask_i]
                cv2.imwrite(save_path, cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))

            pred_sted = (0, 0)
            if args.tmp_loc:
                match = re.search(r"frames:\((\d+),(\d+)\)", text_output)
                if match:
                    pred_t_start = int(match.group(1))
                    pred_t_end = int(match.group(2))
                    pred_sted = (pred_t_start, pred_t_end)
                else:
                    print("No temporal localization predicted.")

            frame_ids = list(range(len(images)))
            gt_sted = (time_s, time_e)
            if gt_sted[0] < 0 or gt_sted[1] < gt_sted[0]:
                inter_frames = []
            else:
                inter_frames = list(range(gt_sted[0], gt_sted[1] + 1))
            video_id = idx

            if args.tmp_loc:
                tiou, union_predgt, inter_predgt = _calc_tiou(gt_sted, pred_sted, frame_ids)
                curr_video_metrics = {
                    "gt_sted": gt_sted,
                    "pred_sted": pred_sted,
                    "tiou": tiou,
                    "qtype": qtype,
                    "img_metrics": {},
                }
                viou = 0.0
            else:
                curr_video_metrics = {"qtype": qtype, "img_metrics": {}}
                union_predgt = frame_ids
                inter_predgt = frame_ids

            gt_viou = 0.0
            for frame_id in inter_frames:
                if frame_id not in predictions:
                    raise RuntimeError(f"No prediction for frame {frame_id}")

                pred_boxes = predictions[frame_id]["boxes"].detach().cpu().numpy()
                gt_boxes = gt_boxes_per_video[frame_id]["boxes"].detach().cpu().numpy()
                iou = np_box_iou(pred_boxes, gt_boxes)[0][0]

                curr_video_metrics["img_metrics"][frame_id] = {
                    "iou": float(iou),
                    "pred_box": pred_boxes[0].tolist(),
                    "gt_box": gt_boxes[0].tolist(),
                }
                if frame_id in inter_predgt and args.tmp_loc:
                    viou += iou
                gt_viou += iou

            if args.tmp_loc:
                viou = viou / max(len(union_predgt), 1)
                curr_video_metrics["viou"] = viou
                recalls = {thresh: 0 for thresh in iou_thresholds}
                for thresh in iou_thresholds:
                    if viou > thresh:
                        recalls[thresh] += 1
                curr_video_metrics.update({f"viou@{thresh}": recalls[thresh] for thresh in iou_thresholds})

            gt_viou = gt_viou / max(len(inter_frames), 1)
            curr_video_metrics["gt_viou"] = gt_viou
            gt_recalls = {thresh: 0 for thresh in iou_thresholds}
            for thresh in iou_thresholds:
                if gt_viou > thresh:
                    gt_recalls[thresh] += 1
            curr_video_metrics.update({f"gt_viou@{thresh}": gt_recalls[thresh] for thresh in iou_thresholds})

            vid_metrics[video_id] = curr_video_metrics
            torch.save(curr_video_metrics, metrics_file)

        except Exception as e:
            print("Error at idx:", idx)
            print("\033[91m\t\t\t", e, "\033[0m")
            continue

    summarize_metrics(vid_metrics, tmp_loc=args.tmp_loc)


if __name__ == "__main__":
    main()
