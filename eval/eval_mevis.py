import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from chat import (
    IMAGE_TOKEN_INDEX,
    TSF_TOKEN_INDEX,
    initialize_model_videogptplus,
    preprocess_tsf_crop,
    preprocess_tsf_from_bbox_txt,
    preprocess_vision,
)
from util.refer_datasets.mevis import MeVISBaseDataset
from util.refer_datasets.new.davis17 import ReferDAVISDataset
from util.refer_datasets.new.ytvos import ReferYouTubeVOSDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run video referring segmentation inference for MeViS/Refer-YouTube-VOS/Refer-DAVIS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--llava_version_or_path", type=str, default="hf_model", help="Path to model directory.")
    parser.add_argument("--vis_save_path", type=str, default="vis_output/eval_mevis", help="Output root for predicted masks.")
    parser.add_argument("--precision", type=str, default="fp16", choices=["bf16", "fp16", "fp32"], help="Inference precision.")
    parser.add_argument("--local_rank", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--load_in_8bit", action="store_true", default=False, help="Load LLM in 8-bit.")
    parser.add_argument("--load_in_4bit", action="store_true", default=False, help="Load LLM in 4-bit.")
    parser.add_argument("--use_sam2_video_branch", action="store_true", default=False, help="Use SAM2 video branch at inference.")
    parser.add_argument("--base_model_type", type=str, default="vgpt|phi3", choices=["vgpt|phi3"], help="Model family.")
    parser.add_argument("--use_tsf_token", action="store_true", default=False, help="Enable TSF token pathway.")
    parser.add_argument("--tsf_k", type=int, default=4, help="Number of TSF representatives after k-means.")
    parser.add_argument("--tsf_kmeans_iters", type=int, default=8, help="TSF k-means iterations.")
    parser.add_argument("--tsf_crop_path", type=str, default="", help="Static TSF crop image path used for all clips.")
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
    parser.add_argument(
        "--dataset_name",
        default="MEVIS|valid",
        type=str,
        choices=[
            "MEVIS|valid",
            "MEVIS|valid_u",
            "ReferYouTubeVOS|valid",
            "ReferYouTubeVOS|test",
            "ReferDAVIS|valid",
        ],
        help="Dataset and split.",
    )
    parser.add_argument("--resume_unprocessed", action="store_true", default=True, help="Skip samples where all mask PNGs already exist.")
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


def _expression_done(output_dir, expected_frames):
    if not os.path.isdir(output_dir):
        return False
    pngs = [f for f in os.listdir(output_dir) if f.endswith(".png")]
    return len(pngs) >= expected_frames


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


def _load_dataset(args):
    dataset_name, split = args.dataset_name.split("|")
    if dataset_name == "MEVIS":
        dataset = MeVISBaseDataset(args.video_dataset_dir, image_set=split, num_frames=-1)
    elif dataset_name == "ReferYouTubeVOS":
        dataset = ReferYouTubeVOSDataset(args.video_dataset_dir, split=split)
    elif dataset_name == "ReferDAVIS":
        dataset = ReferDAVISDataset(args.video_dataset_dir, split=split)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")
    return dataset_name, split, dataset


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

    dataset_name, split, eval_dataset = _load_dataset(args)

    autocast_enabled = args.precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    for idx in tqdm(range(len(eval_dataset)), desc="Processing samples"):
        try:
            np_images, target = eval_dataset[idx]
            video_name, exp_id = target["video_path"]
            gt_caption = target["caption"]
            frame_ids = target.get("frame_ids")

            video_frames_np = [np_images[t] for t in range(np_images.shape[0])]
            video_len = len(video_frames_np)
            np_images_batch = [video_frames_np]

            output_dir = os.path.join(
                args.vis_save_path,
                f"{dataset_name}____{split}_output",
                video_name,
                exp_id,
            )
            if args.resume_unprocessed and _expression_done(output_dir, expected_frames=video_len):
                continue
            os.makedirs(output_dir, exist_ok=True)

            prompt_text = f"What is {gt_caption.lower()} in this video? Please respond with segmentation masks."
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
                tsf_boxes_path = _resolve_tsf_boxes_path(args.tsf_boxes_txt, video_name)
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
            if args.print_text_output:
                text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
                text_output = text_output.replace("\n", "").replace("  ", " ").split("ASSISTANT: ")[-1]
                print(f"text_output ({video_name}, {exp_id}): {text_output}")

            video_segments = video_segments_batch[0]
            frame_h, frame_w = video_frames_np[0].shape[:2]
            for t in range(video_len):
                frame_preds = video_segments.get(t, {}) if isinstance(video_segments, dict) else {}
                pred_mask_i = _first_mask_for_frame(frame_preds, target_hw=(frame_h, frame_w))

                if frame_ids is not None and dataset_name in {"ReferYouTubeVOS", "ReferDAVIS"} and t < len(frame_ids):
                    mask_name = f"{frame_ids[t]}.png"
                else:
                    mask_name = f"{t:05d}.png"

                mask_img = Image.fromarray((pred_mask_i.astype(np.uint8) * 255))
                mask_img.save(os.path.join(output_dir, mask_name))

        except Exception as e:
            print("Error at idx:", idx)
            print("\033[91m\t\t\t", e, "\033[0m")
            continue


if __name__ == "__main__":
    main()
