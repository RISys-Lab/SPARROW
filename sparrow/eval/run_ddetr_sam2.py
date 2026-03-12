import os
import copy
import torch
import argparse
import requests
from PIL import Image, ImageDraw
from io import BytesIO
from torchvision import transforms
from transformers.image_transforms import center_to_corners_format

from transformers import DeformableDetrConfig
from sparrow.utils import disable_torch_init


def _nms(boxes, scores, iou_threshold):
    try:
        from mmcv.ops.nms import nms
    except ImportError as exc:
        raise ImportError(
            "mmcv is required for inference NMS. Please install mmcv/mmcv-full matching your PyTorch/CUDA."
        ) from exc
    return nms(boxes, scores, iou_threshold)[-1]


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA device requested, but CUDA is not available.")
    return device_arg


def preprocess_image(raw_image, image_size):
    # Match training-time normalization used by det datasets.
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(raw_image).unsqueeze(0)


def build_model(
    model_name,
    sam2_ckpt,
    sam2_cfg,
    num_queries,
    num_classes,
    num_feature_levels,
    device,
):
    from sparrow.model.ddetr_sam2 import CustomDDETRSAM2Model
    from sparrow.model.ddetr_sam2 import CustomDDETRSAM2Config

    if model_name is not None:
        model = CustomDDETRSAM2Model.from_pretrained(model_name).to(device)
        return model

    ddetr_cfg = DeformableDetrConfig(
        num_queries=num_queries,
        two_stage=True,
        two_stage_num_proposals=num_queries,
        num_labels=num_classes,
        with_box_refine=True,
        num_feature_levels=num_feature_levels,
    )
    model_cfg = CustomDDETRSAM2Config(
        sam2_cfg=sam2_cfg,
        ddetr_cfg=ddetr_cfg,
    )
    model = CustomDDETRSAM2Model(model_cfg, pretrained_vis_encoder=sam2_ckpt).to(device)
    return model


def eval_model(
    model_name,
    sam2_ckpt,
    sam2_cfg,
    image_file,
    image_size,
    num_queries,
    num_classes,
    num_feature_levels,
    device,
    output_dir,
    nms_threshold,
    score_threshold,
    coco_score_power,
    sa1b_score_power,
):
    disable_torch_init()
    if model_name is not None:
        model_name = os.path.expanduser(model_name)
    if sam2_ckpt is not None:
        sam2_ckpt = os.path.expanduser(sam2_ckpt)
    image_file = os.path.expanduser(image_file)

    model = build_model(
        model_name=model_name,
        sam2_ckpt=sam2_ckpt,
        sam2_cfg=sam2_cfg,
        num_queries=num_queries,
        num_classes=num_classes,
        num_feature_levels=num_feature_levels,
        device=device,
    )
    model.eval()

    raw_image = load_image(image_file)
    resized_image = raw_image.resize((image_size, image_size))
    image = preprocess_image(raw_image, image_size).to(device)

    with torch.inference_mode():
        outputs = model(image)

    pred_boxes = outputs.pred_boxes
    pred_boxes = center_to_corners_format(pred_boxes)
    scores_coco = outputs.logits['coco'].squeeze().sigmoid()
    scores_sa1b = outputs.logits['sa1b'].squeeze().sigmoid()

    nms_inds = _nms(pred_boxes[0], scores_coco + scores_sa1b, nms_threshold)
    thres_scores_comb = [
        i for i in range(len(scores_coco))
        if scores_coco[i] ** coco_score_power * scores_sa1b[i] ** sa1b_score_power >= score_threshold and i in nms_inds
    ]

    os.makedirs(output_dir, exist_ok=True)

    img_copy = copy.deepcopy(resized_image)
    for box in pred_boxes[0, thres_scores_comb]:
        w, h = img_copy.size
        box = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
        draw = ImageDraw.Draw(img_copy)
        draw.rectangle(box, outline="red")

    img_name = os.path.splitext(os.path.basename(image_file))[0]
    output_file = os.path.join(output_dir, f"{img_name}_filter.jpg")
    img_copy.save(output_file, "JPEG")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SAM2 DDETR inference and save box visualization images."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Path to a pretrained ddetr_sam2 checkpoint directory. If set, this takes priority.",
    )
    parser.add_argument(
        "--sam2-ckpt",
        type=str,
        default=None,
        help="SAM2 checkpoint path used when --model-name is not provided.",
    )
    parser.add_argument("--sam2-cfg", type=str, default="sam2_hiera_l.yaml", help="SAM2 Hydra config name.")
    img_group = parser.add_mutually_exclusive_group(required=True)
    img_group.add_argument("--image-dir", type=str, default=None, help="Directory of images to run inference on.")
    img_group.add_argument("--image-file", type=str, default=None, help="Single image path or URL.")
    parser.add_argument("--output-dir", type=str, default="det_vis", help="Where to save visualized predictions.")
    parser.add_argument("--device", type=str, default="auto", help='Device: "auto", "cuda", "cuda:0", or "cpu".')
    parser.add_argument("--image-size", type=int, default=448, help="Input resize (square).")
    parser.add_argument("--num-queries", type=int, default=300)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--num-feature-levels", type=int, default=1)
    parser.add_argument("--nms-threshold", type=float, default=0.8)
    parser.add_argument("--score-threshold", type=float, default=0.4)
    parser.add_argument("--coco-score-power", type=float, default=0.3)
    parser.add_argument("--sa1b-score-power", type=float, default=0.7)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    if args.model_name is None and args.sam2_ckpt is None:
        print("No --model-name provided. Running with random SAM2 + random DDETR weights.")
    elif args.model_name is None and args.sam2_ckpt is not None:
        print("No --model-name provided. Running with SAM2 checkpoint + random DDETR weights.")
    elif args.model_name is not None and args.sam2_ckpt is not None:
        print("Both --model-name and --sam2-ckpt were provided. --model-name will be used.")
    elif args.model_name is not None:
        print("Running with pretrained ddetr_sam2 checkpoint from --model-name.")

    if args.image_dir is not None:
        image_dir = os.path.expanduser(args.image_dir)
        image_files = sorted(
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if len(image_files) == 0:
            raise ValueError(f"No supported images found in directory: {image_dir}")
        for image_name in image_files:
            image_file = os.path.join(image_dir, image_name)
            eval_model(
                args.model_name,
                args.sam2_ckpt,
                args.sam2_cfg,
                image_file,
                args.image_size,
                args.num_queries,
                args.num_classes,
                args.num_feature_levels,
                device,
                args.output_dir,
                args.nms_threshold,
                args.score_threshold,
                args.coco_score_power,
                args.sa1b_score_power,
            )
    else:
        eval_model(
            args.model_name,
            args.sam2_ckpt,
            args.sam2_cfg,
            args.image_file,
            args.image_size,
            args.num_queries,
            args.num_classes,
            args.num_feature_levels,
            device,
            args.output_dir,
            args.nms_threshold,
            args.score_threshold,
            args.coco_score_power,
            args.sa1b_score_power,
        )
