import argparse
import json
import os
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Refer-DAVIS inference and save DAVIS-style indexed PNG masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--llava_version_or_path", type=str, default="hf_model", help="Path to model directory.")
    parser.add_argument("--vis_save_path", type=str, default="vis_output/eval_davis17", help="Output root for predicted masks.")
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
        help="Dataset root containing processed/refer_davis/2017.",
    )
    parser.add_argument("--split", type=str, default="valid", choices=["valid"], help="Refer-DAVIS split.")
    parser.add_argument("--resume_unprocessed", action="store_true", default=True, help="Skip annotation/video folders that already have all frames.")
    parser.add_argument("--clip_size", type=int, default=64, help="Temporal chunk size for inference.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new text tokens during generation.")
    parser.add_argument("--print_text_output", action="store_true", default=False, help="Print decoded model text for each clip.")
    return parser.parse_args()


def _annotation_done(save_path_prefix, anno_name, video_name, expected_frames):
    anno_save_path = os.path.join(save_path_prefix, anno_name, video_name)
    if not os.path.isdir(anno_save_path):
        return False
    pngs = [f for f in os.listdir(anno_save_path) if f.endswith(".png")]
    return len(pngs) >= expected_frames


def _segments_to_pred_masks(video_segments, clip_len, orig_hw):
    h, w = int(orig_hw[0]), int(orig_hw[1])
    pred_masks = torch.zeros((clip_len, h, w), dtype=torch.float32)
    if not isinstance(video_segments, dict) or len(video_segments) == 0:
        return pred_masks

    for t in range(clip_len):
        frame_preds = video_segments.get(t, {})
        if not isinstance(frame_preds, dict) or len(frame_preds) == 0:
            continue
        first_obj_id = sorted(frame_preds.keys())[0]
        mask_t = torch.as_tensor(frame_preds[first_obj_id], dtype=torch.float32)
        if mask_t.ndim == 3:
            mask_t = mask_t[0]
        if tuple(mask_t.shape[-2:]) != (h, w):
            mask_t = F.interpolate(mask_t.unsqueeze(0).unsqueeze(0), size=(h, w), mode="nearest")[0, 0]
        pred_masks[t] = mask_t
    return pred_masks


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

    davis_path = os.path.join(args.video_dataset_dir, "processed/refer_davis/2017")
    split = args.split
    output_dir = args.vis_save_path

    palette_img = os.path.join(davis_path, "valid/Annotations/blackswan/00000.png")
    palette = Image.open(palette_img).getpalette()

    root = Path(davis_path)
    img_folder = os.path.join(root, split, "JPEGImages")
    meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    video_list = list(data.keys())

    save_path_prefix = os.path.join(output_dir, split)
    os.makedirs(save_path_prefix, exist_ok=True)

    save_name_expression = {
        0: "Davis17_annot1",
        1: "Davis17_annot1_full_video",
        2: "Davis17_annot2",
        3: "Davis17_annot2_full_video",
    }

    autocast_enabled = args.precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    for video in tqdm(video_list, desc="Processing videos"):
        expressions = data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        video_len = len(data[video]["frames"])

        metas = []
        for exp_idx in range(num_expressions):
            metas.append(
                {
                    "video": video,
                    "exp": expressions[expression_list[exp_idx]]["exp"],
                    "exp_id": expression_list[exp_idx],
                    "frames": data[video]["frames"],
                }
            )

        num_obj = num_expressions // 4
        for anno_id in range(4):
            if args.resume_unprocessed and _annotation_done(
                save_path_prefix=save_path_prefix,
                anno_name=save_name_expression[anno_id],
                video_name=video,
                expected_frames=video_len,
            ):
                continue

            anno_masks = []
            for obj_id in range(num_obj):
                meta_idx = obj_id * 4 + anno_id
                video_name = metas[meta_idx]["video"]
                exp = metas[meta_idx]["exp"]
                frames = metas[meta_idx]["frames"]
                curr_video_len = len(frames)

                all_pred_masks = []
                for clip_id in range(0, curr_video_len, args.clip_size):
                    clip_frames_ids = range(clip_id, min(clip_id + args.clip_size, curr_video_len))
                    imgs = [
                        Image.open(os.path.join(img_folder, video_name, frames[t] + ".jpg")).convert("RGB")
                        for t in clip_frames_ids
                    ]
                    np_images = [[np.array(img) for img in imgs]]

                    prompt_text = f"What is {exp.lower()} in this video? Please respond with segmentation masks."
                    enc_image, enc_context_image, image_sam, original_size_list, resize_list = preprocess_vision(
                        np_images=np_images,
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
                                np_images=np_images,
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
                        print(f"text_output ({video_name}, anno={anno_id}, obj={obj_id}, clip={clip_id}): {text_output}")

                    pred_masks = _segments_to_pred_masks(
                        video_segments=video_segments_batch[0],
                        clip_len=len(clip_frames_ids),
                        orig_hw=original_size_list[0],
                    )
                    all_pred_masks.append(pred_masks)

                anno_masks.append(torch.cat(all_pred_masks, dim=0))

            anno_masks = torch.stack(anno_masks)  # [num_obj, T, H, W]
            t, h, w = anno_masks.shape[-3:]
            anno_masks[anno_masks < 0.5] = 0.0
            background = 0.1 * torch.ones(1, t, h, w, device=anno_masks.device)
            anno_masks = torch.cat([background, anno_masks], dim=0)  # [num_obj+1, T, H, W]
            out_masks = torch.argmax(anno_masks, dim=0).detach().cpu().numpy().astype(np.uint8)

            anno_save_path = os.path.join(save_path_prefix, save_name_expression[anno_id], video)
            os.makedirs(anno_save_path, exist_ok=True)
            for frame_idx in range(out_masks.shape[0]):
                img_e = Image.fromarray(out_masks[frame_idx])
                img_e.putpalette(palette)
                img_e.save(os.path.join(anno_save_path, f"{frame_idx:05d}.png"))


if __name__ == "__main__":
    main()
