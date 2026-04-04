import argparse
import json
import os
import re

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
from util.dataset import ValGCGDataset


def remove_small_blobs(binary_mask: np.ndarray, min_size: int = 0):
    if min_size > 0:
        dtype = binary_mask.dtype
        binary_mask = skimage.morphology.remove_small_objects(binary_mask.astype(bool), min_size=min_size)
        binary_mask = binary_mask.astype(dtype)
    return binary_mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GCG inference and save per-video outputs for downstream metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--llava_version_or_path", type=str, default="hf_model", help="Path to model directory.")
    parser.add_argument("--vis_save_path", type=str, default="vis_output/eval_gcg_new", help="Output root for predicted masks.")
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
    parser.add_argument("--dataset_name", default="video_gcg", type=str, choices=["video_gcg"], help="GCG validation set.")
    parser.add_argument("--resume_unprocessed", action="store_true", default=True, help="Skip samples where res.json already exists.")
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


def _mask_to_bool(mask_t, target_hw):
    h, w = int(target_hw[0]), int(target_hw[1])
    mask_t = torch.as_tensor(mask_t, dtype=torch.float32)
    if mask_t.ndim == 3:
        mask_t = mask_t[0]
    if mask_t.ndim != 2:
        return np.zeros((h, w), dtype=np.bool_)
    if tuple(mask_t.shape[-2:]) != (h, w):
        mask_t = F.interpolate(mask_t.unsqueeze(0).unsqueeze(0), size=(h, w), mode="nearest")[0, 0]
    return (mask_t > 0).detach().cpu().numpy().astype(np.bool_)


def clean_caption(text_output):
    text_output_ = text_output.replace("\n", "").replace("  ", " ")
    text_output_ = text_output_.split("ASSISTANT: ")[-1]
    cleaned_str = re.sub(r"<.*?>", "", text_output_)
    pattern = re.compile(r"<p>(.*?)<\/p>")
    phrases = [p.strip() for p in pattern.findall(text_output_)]
    cleaned_str = cleaned_str.replace("[SEG]", "")
    cleaned_str = " ".join(cleaned_str.split()).strip("'")
    cleaned_str = cleaned_str.strip()
    return cleaned_str, phrases


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

    eval_dataset = ValGCGDataset(args.video_dataset_dir, val_datasets="video_gcg||mevis_gcg||vidstg_gcg")

    autocast_enabled = args.precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    for idx in tqdm(range(len(eval_dataset)), desc="Processing samples"):
        try:
            save_dir_for_current_video = os.path.join(args.vis_save_path, args.dataset_name, f"{idx:06d}")
            os.makedirs(save_dir_for_current_video, exist_ok=True)
            saved_file = os.path.join(save_dir_for_current_video, "res.json")
            if args.resume_unprocessed and os.path.exists(saved_file):
                print(f"Skipping {idx} as it already exists.")
                continue

            video_name, _, pil_images, gt_masks, gt_caption, _ = eval_dataset[idx]

            res = {
                "gt_text": gt_caption,
                "gt_masks": gt_masks,
            }
            res["gt_text_cleaned"], res["gt_phrases"] = clean_caption(gt_caption)

            np_images = [np.array(image) for image in pil_images]
            np_images_batch = [np_images]

            prompt_text = (
                "Could you please give me a detailed description of the video? "
                "Please respond with interleaved segmentation masks for the corresponding parts of the answer."
            )
            enc_image, enc_context_image, image_sam, original_size_list, resize_list = preprocess_vision(
                np_images=np_images_batch,
                type="video",
                enc_preprocessor=enc_preprocessor,
                sam_preprocessor=sam_preprocessor,
                conv_generator=conv_generator,
                precision=args.precision,
            )
            input_ids = conv_generator.apply_for_chat(prompt_text, type="video", tokenizer=tokenizer)

            tsf_images = static_tsf_images
            if args.use_tsf_token and tsf_images is None and args.tsf_boxes_txt:
                tsf_boxes_path = _resolve_tsf_boxes_path(args.tsf_boxes_txt, str(video_name))
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
            text_output = text_output.replace("\n", "").replace("  ", " ").split("ASSISTANT: ")[-1]
            if args.print_text_output:
                print(f"text_output ({video_name}): {text_output}")

            res["pred_text"] = text_output
            res["pred_text_cleaned"], res["pred_phrases"] = clean_caption(text_output)
            res["img_frames"] = {t: np_images[t] for t in range(len(np_images))}

            video_segments = video_segments_batch[0]
            frame_h, frame_w = np_images[0].shape[:2]
            num_frames = len(np_images)

            obj_ids = set()
            if isinstance(video_segments, dict):
                for frame_preds in video_segments.values():
                    if isinstance(frame_preds, dict):
                        obj_ids.update(frame_preds.keys())

            pred_masks = {
                obj_id: [np.zeros((frame_h, frame_w), dtype=np.bool_) for _ in range(num_frames)]
                for obj_id in sorted(obj_ids)
            }
            for t in range(num_frames):
                frame_preds = video_segments.get(t, {}) if isinstance(video_segments, dict) else {}
                if not isinstance(frame_preds, dict):
                    continue
                for obj_id, pred_mask_i in frame_preds.items():
                    pred_mask_i = _mask_to_bool(pred_mask_i, target_hw=(frame_h, frame_w))
                    pred_mask_i = remove_small_blobs(pred_mask_i, min_size=20)
                    if obj_id not in pred_masks:
                        pred_masks[obj_id] = [np.zeros((frame_h, frame_w), dtype=np.bool_) for _ in range(num_frames)]
                    pred_masks[obj_id][t] = pred_mask_i

            res["pred_masks"] = {obj_id: np.stack(masks, axis=0) for obj_id, masks in pred_masks.items()}

            res_to_save = {
                "gt_text": res["gt_text"],
                "gt_text_cleaned": res["gt_text_cleaned"],
                "gt_phrases": res["gt_phrases"],
                "pred_text": res["pred_text"],
                "pred_text_cleaned": res["pred_text_cleaned"],
                "pred_phrases": res["pred_phrases"],
            }
            with open(saved_file, "w") as f:
                json.dump(res_to_save, f)

            img_frames_dir = os.path.join(save_dir_for_current_video, "img_frames")
            os.makedirs(img_frames_dir, exist_ok=True)
            for frame_idx, image in res["img_frames"].items():
                image_path = os.path.join(img_frames_dir, f"frame_{frame_idx}.jpg")
                cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            for obj_id, masks in res["gt_masks"].items():
                obj_dir = os.path.join(save_dir_for_current_video, f"gt_masks_{obj_id}")
                os.makedirs(obj_dir, exist_ok=True)
                for i, mask in enumerate(masks):
                    mask_path = os.path.join(obj_dir, f"mask_{i}.png")
                    skimage.io.imsave(mask_path, skimage.img_as_ubyte(mask.astype(np.bool_)), check_contrast=False)

            for obj_id, masks in res["pred_masks"].items():
                obj_dir = os.path.join(save_dir_for_current_video, f"pred_masks_{obj_id}")
                os.makedirs(obj_dir, exist_ok=True)
                for i, mask in enumerate(masks):
                    mask_path = os.path.join(obj_dir, f"mask_{i}.png")
                    skimage.io.imsave(mask_path, skimage.img_as_ubyte(mask.astype(np.bool_)), check_contrast=False)

            print(f"Saved idx:{idx} to {save_dir_for_current_video}")

        except Exception as e:
            print("Error at idx:", idx)
            print("\033[91m\t\t\t", e, "\033[0m")
            continue


if __name__ == "__main__":
    main()
