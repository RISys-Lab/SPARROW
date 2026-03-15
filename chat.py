import argparse
import os
import re
import shutil
import cv2
import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

import transformers
from transformers import AutoTokenizer

from decord import VideoReader, cpu
from PIL import Image

DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_TOKEN_INDEX = -200
TSF_TOKEN_INDEX = -201
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
SAM_PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
SAM_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)


def write_masks(video_segments, video_frames_np, save_dir):
    ''' Write masks to disk 
    Args:
    - video_segments: dictionary with keys being frame indices, and values being dictionaries with keys being segment indices
    - video_frames_np: numpy array of video frames # [T, H, W, C]  # numpy array
    '''
    
    # video_segments is a dictionary with keys being frame indices
    # video_segments[t] is a dictionary with keys being segment indices
    
    for t, pred_mask in video_segments.items():
        
        # save image frame
        save_img = video_frames_np[t].copy()
        img_dir = os.path.join(save_dir, "img_frames")
        os.makedirs(img_dir, exist_ok=True)
        img_save_path = os.path.join(img_dir, f"frame_{t}.jpg")
        cv2.imwrite(img_save_path, cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))        
        
        # save mask for each object
        for obj_id, pred_mask_i in pred_mask.items():
            pred_mask_i = pred_mask_i > 0

            # save mask
            obj_dir = os.path.join(save_dir, f"pred_masks_{obj_id}")
            os.makedirs(obj_dir, exist_ok=True)
            mask_path = os.path.join(obj_dir, f"mask_{t}.png")
            cv2.imwrite(mask_path, pred_mask_i * 255)
            print("{} has been saved.".format(mask_path))
            
            # save masked image frame
            save_path = "{}/masked_images/masked_img_{}_{}.jpg".format(save_dir, t, obj_id)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            save_img = video_frames_np[t].copy()
            save_img[pred_mask_i] = (video_frames_np[t] * 0.5 + pred_mask_i[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5)[pred_mask_i]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            print("{} has been saved.".format(save_path))

def _get_rawvideo_dec(video_path, video_framerate=1, s=None, e=None):
    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0. else 0.
        end_time = end_time if end_time >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))
        sample_pos = list(range(f_start, f_end + 1, t_stride))
        np_images = [f for f in vreader.get_batch(sample_pos).asnumpy()]
    else:
        print("video path: {} error.".format(video_path))

    return np_images

def get_args():
    parser = argparse.ArgumentParser(
        description="SPARROW chat inference for image/video segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--llava_version_or_path", type=str, default="hf_model", help="Path to HF model directory.")
    parser.add_argument("--input_path", type=str, required=True, help="Input image/video path.")
    parser.add_argument("--prompt_text", type=str, required=True, help="Prompt text used for generation.")
    parser.add_argument("--vis_save_path", type=str, default="vis_output/chat_output", help="Root output directory.")
    parser.add_argument("--precision", type=str, default="fp16", choices=["bf16", "fp16", "fp32"], help="Inference precision.")
    parser.add_argument("--local_rank", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum new generated tokens.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load LLM in 8-bit mode.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load LLM in 4-bit mode.")

    parser.add_argument("--use_tsf_token", default=False, action="store_true", help="Enable TSF token usage.")
    parser.add_argument("--tsf_k", type=int, default=4, help="Number of TSF representatives after k-means.")
    parser.add_argument("--tsf_kmeans_iters", type=int, default=8, help="Number of TSF k-means iterations.")
    parser.add_argument("--tsf_crop_path", type=str, default="", help="Single TSF crop image path.")
    parser.add_argument("--tsf_boxes_txt", type=str, default="", help="BBox txt for TSF crop extraction fallback.")

    parser.add_argument("--use_sam2_video_branch", action="store_true", help="Use SAM2 video-branch for the main run.")
    parser.add_argument(
        "--proposal_debug_modes",
        type=str,
        default="both",
        choices=["selected", "both"],
        help="Save proposal visualizations for the selected branch only or both branch modes.",
    )

    parser.add_argument("--base_model_type", type=str, default="vgpt|phi3", choices=["vgpt|phi3"], help="Backbone type.")
    parser.add_argument("--ddetr_model_path", type=str, default="checkpoints_hf/ddetr_sam2", help="DDETR model directory.")
    parser.add_argument("--proposal_topk", type=int, default=100, help="Number of detector proposals.")
    parser.add_argument("--proposal_score_power", type=float, nargs=2, default=(0.3, 0.7), help="DDETR score fusion exponents.")
    parser.add_argument("--proposal_score_threshold", type=float, default=0.4, help="Selection threshold on proposal logits.")
    parser.add_argument("--use_proposal_box_regression", action="store_true", default=False, help="Enable proposal box regression branch.")
    parser.add_argument("--proposal_box_loss_weight", type=float, default=0.01, help="Weight for proposal box regression branch.")
    parser.add_argument("--roi_helper_hidden_dim", type=int, default=256, help="ROI helper hidden dimension.")
    parser.add_argument("--roi_align_output_size", type=int, default=7, help="ROIAlign output size.")
    parser.add_argument("--roi_fusion_init", type=float, default=0.1, help="Initial fusion scalar for ROI helper logits.")
    parser.add_argument("--box_prior_min_score", type=float, default=0.35, help="Minimum prior score for keeping a selected box.")
    parser.add_argument("--box_prior_max_keep", type=int, default=4, help="Maximum selected boxes per token after prior filtering.")
    parser.add_argument("--save_proposal_debug", action="store_true", default=True, help="Save proposal overlays before/after filtering.")
    parser.add_argument("--no_save_proposal_debug", dest="save_proposal_debug", action="store_false", help="Disable proposal debug saving.")

    parser.set_defaults(
        use_detection_guidance=True,
        use_roi_helper=True,
        use_box_prior_filter=True,
    )
    parser.add_argument("--use_detection_guidance", dest="use_detection_guidance", action="store_true", help="Enable detection guidance.")
    parser.add_argument("--disable_detection_guidance", dest="use_detection_guidance", action="store_false", help="Disable detection guidance.")
    parser.add_argument("--use_roi_helper", dest="use_roi_helper", action="store_true", help="Enable ROI helper head.")
    parser.add_argument("--disable_roi_helper", dest="use_roi_helper", action="store_false", help="Disable ROI helper head.")
    parser.add_argument("--use_box_prior_filter", dest="use_box_prior_filter", action="store_true", help="Enable prior-based box filtering.")
    parser.add_argument("--disable_box_prior_filter", dest="use_box_prior_filter", action="store_false", help="Disable prior-based box filtering.")

    return parser.parse_args()

def initialize_model_videogptplus(
    model_base,
    precision,
    local_rank,
    load_in_8bit,
    load_in_4bit,
    use_sam2_video_branch,
    base_llm_type,
    use_detection_guidance=False,
    ddetr_model_path=None,
    proposal_topk=100,
    proposal_score_power=(0.3, 0.7),
    proposal_score_threshold=0.4,
    use_proposal_box_regression=False,
    proposal_box_loss_weight=0.01,
    use_roi_helper=True,
    roi_helper_hidden_dim=256,
    roi_align_output_size=7,
    roi_fusion_init=0.1,
    use_box_prior_filter=False,
    box_prior_min_score=0.35,
    box_prior_max_keep=4,
    use_tsf_token=False,
    tsf_k=4,
    tsf_kmeans_iters=8,
):
    # model_args
    torch_dtype = torch.bfloat16 if precision == "bf16" else (torch.half if precision == "fp16" else torch.float32)
    model_args = {"torch_dtype": torch_dtype}
    if load_in_4bit:
        model_args.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif load_in_8bit:
        model_args.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": transformers.BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    # Load model
    from model.SPARROW import SPARROWForCausalLM
    model_args.update(
        {
            "use_detection_guidance": use_detection_guidance,
            "ddetr_model_path": ddetr_model_path,
            "proposal_topk": proposal_topk,
            "proposal_score_power": tuple(proposal_score_power) if proposal_score_power is not None else (0.3, 0.7),
            "proposal_score_threshold": proposal_score_threshold,
            "use_proposal_box_regression": use_proposal_box_regression,
            "proposal_box_loss_weight": proposal_box_loss_weight,
            "use_roi_helper": use_roi_helper,
            "roi_helper_hidden_dim": roi_helper_hidden_dim,
            "roi_align_output_size": roi_align_output_size,
            "roi_fusion_init": roi_fusion_init,
            "use_box_prior_filter": use_box_prior_filter,
            "box_prior_min_score": box_prior_min_score,
            "box_prior_max_keep": box_prior_max_keep,
        }
    )
    if base_llm_type == "phi3":
        model = SPARROWForCausalLM.from_pretrained(
            model_base, low_cpu_mem_usage=False, 
            use_sam2_video_branch=use_sam2_video_branch,
            **model_args)
    else:
        raise ValueError("Invalid base_llm_type")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    # Add new tokens to tokenizer
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", False)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    tokenizer.add_tokens("[BOX]")
    seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")
    box_token_idx = tokenizer.convert_tokens_to_ids("[BOX]")
    print("seg_token_idx: ", seg_token_idx)
    print("box_token_idx: ", box_token_idx)
    model.resize_token_embeddings(len(tokenizer))
    
    # set seg_token_idx in model config
    model.config.seg_token_idx = seg_token_idx
    model.config.box_token_idx = box_token_idx
    # set model configs
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_tsf_token = use_tsf_token
    model.config.tsf_k = int(tsf_k)
    model.config.tsf_kmeans_iters = int(tsf_kmeans_iters)
    
    # for llama3_1
    if tokenizer.pad_token_id == None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Initialize encoder modules
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=local_rank)
    image_vision_tower = model.get_model().get_image_vision_tower()
    image_vision_tower.to(dtype=torch_dtype, device=local_rank)

    ### Model dtype
    if precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        precision == "fp16" and (not load_in_4bit) and (not load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        image_vision_tower = model.get_model().get_image_vision_tower()
        model.model.image_vision_tower = None
        
        import deepspeed
        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=False, #NOTE
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
        model.model.image_vision_tower = image_vision_tower.half().cuda()
    elif precision == "fp32":
        model = model.float().cuda()
        
    model.eval()
        
    # enc_preprocessor
    from util.enc_preprocessors import EncPreprocessor_VideoGPTPlus
    enc_preprocessor = EncPreprocessor_VideoGPTPlus()
    
    # conversation_generator
    from util.conv_generator import ConvGenerator_VideoGPTPlus
    conv_generator = ConvGenerator_VideoGPTPlus(
        use_mm_start_end=mm_use_im_start_end,
        base_type=base_llm_type,
        use_tsf_token=use_tsf_token,
    )
    
    # sam preprocessor
    if model.config.use_sam2:
        from util.sam_transforms import SAM_v2_Preprocess
        sam_preprocessor = SAM_v2_Preprocess()
    else:
        from util.sam_transforms import SAM_v1_Preprocess
        sam_preprocessor = SAM_v1_Preprocess()
    
    return model, tokenizer, enc_preprocessor, conv_generator, sam_preprocessor

def load_image(image_path):
    ''' Returns: numpy array of image # B x T x (H x W x C)
    '''
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    image_np= [[image_np]] # B x T x (H x W x C)
    
    return image_np

def load_video(video_path):
    ''' Returns: list of numpy arrays of video frames # T x (H x W x C) '''
    ### Load video
    video_framerate = 1
    max_num_frames = 64
    np_images = _get_rawvideo_dec(
        video_path, 
        video_framerate=video_framerate, 
        s=None, e=None)
    
    if len(np_images) > max_num_frames:
        new_np_images_idxs = np.linspace(0, len(np_images)-1, max_num_frames, dtype=int)
        new_np_images = [np_images[i] for i in new_np_images_idxs]
        np_images = new_np_images
    
    np_images = [np_images] # B x T x (H x W x C) # Add batch dimension
    
    return np_images
    

def preprocess_vision(np_images, type="video", enc_preprocessor=None, sam_preprocessor=None, conv_generator=None, precision="fp16"):

    if type == "video":
        
        assert len(np_images) == 1, "Batch size must be 1"
        np_images_ = np_images[0] # T x (H x W x C)
        
        # Subsample video frames if longer than max_num_frames
        max_num_frames = conv_generator.NUM_FRAMES
        if len(np_images_) > max_num_frames:
            new_np_images_idxs = np.linspace(0, len(np_images_)-1, max_num_frames, dtype=int)
            np_images_for_enc = [np_images_[i] for i in new_np_images_idxs]
        else:
            np_images_for_enc = np_images_
        
        # Preprocess image for encoder
        pil_images = [Image.fromarray(img) for img in np_images_for_enc]
        image_enc_dict = enc_preprocessor.preprocess(pil_images) # Tx(3 x 224 x 224)
        enc_image = image_enc_dict['images']
        enc_image = torch.stack(enc_image, dim=0) # (T x 3 x 224 x 224)
        enc_image = enc_image.bfloat16() if precision == "bf16" else (enc_image.half() if precision == "fp16" else enc_image.float())
        enc_image = enc_image.cuda(non_blocking=True)
        enc_image = [enc_image] # 1x(T x 3 x 224 x 224) # Add batch dimension
        
        enc_context_image = image_enc_dict['context_images']
        if enc_context_image is not None:
            enc_context_image = torch.stack(enc_context_image, dim=0) # (T x 3 x 224 x 224)
            enc_context_image = enc_context_image.bfloat16() if precision == "bf16" else (enc_context_image.half() if precision == "fp16" else enc_context_image.float())
            enc_context_image = enc_context_image.cuda(non_blocking=True)
            enc_context_image = [enc_context_image] # 1x(T x 3 x 224 x 224) # Add batch dimension
            
        
        ### Preprocess image for SAM
        original_size_list = [np_images_[0].shape[:2]]
        preprocessed_for_sam_and_resize_shapes = [sam_preprocessor.preprocess(image) for image in np_images_]
        image_sam = [x[0] for x in preprocessed_for_sam_and_resize_shapes]
        resize_shape = preprocessed_for_sam_and_resize_shapes[0][1]
        resize_list = [resize_shape]
        image_sam = torch.stack(image_sam, dim=0).cuda() # (T x 3 x 1024 x 1024)
        
        image_sam = image_sam.bfloat16() if precision == "bf16" else (image_sam.half() if precision == "fp16" else image_sam.float())
        image_sam = image_sam.cuda(non_blocking=True)
        image_sam = [image_sam] # 1x (T x 3 x 1024 x 1024) # Add batch dimension
        
    elif type == "image":
        
        assert len(np_images) == 1, "Batch size must be 1"
        assert len(np_images[0]) == 1, "Time dimension must be 1"
        image_np = np_images[0][0] # (H x W x C)
        
        # Preprocess image for encoder
        image_enc_dict = enc_preprocessor.preprocess(image_np) # (3, 224, 224)
        enc_image = image_enc_dict['images']
        enc_image = enc_image.unsqueeze(0) # (1, 3, 224, 224) # Add time dimension
        enc_image = enc_image.bfloat16() if precision == "bf16" else (enc_image.half() if precision == "fp16" else enc_image.float())
        enc_image = enc_image.cuda(non_blocking=True)
        enc_image = [enc_image] # 1x(1, 3, 224, 224) # Add batch dimension
        
        enc_context_image = image_enc_dict['context_images']
        if enc_context_image is not None:
            enc_context_image = enc_context_image.unsqueeze(0) # (1, 3, 224, 224) # Add time dimension
            enc_context_image = enc_context_image.bfloat16() if precision == "bf16" else (enc_context_image.half() if precision == "fp16" else enc_context_image.float())
            enc_context_image = enc_context_image.cuda(non_blocking=True)
            enc_context_image = [enc_context_image] # 1x(1, 3, 224, 224) # Add batch dimension
        
        ### Preprocess image for SAM
        original_size_list = [image_np.shape[:2]]
        
        image_sam, resize_shape = sam_preprocessor.preprocess(image_np)
        resize_list = [resize_shape]
        
        image_sam = image_sam.bfloat16() if precision == "bf16" else (image_sam.half() if precision == "fp16" else image_sam.float())
        image_sam = image_sam.unsqueeze(0).cuda() # (1, 3, 1024, 1024) # Add time dimension
        image_sam = [image_sam] # 1x(1, 3, 1024, 1024) # Add batch dimension
        
    return enc_image, enc_context_image, image_sam, original_size_list, resize_list


def preprocess_tsf_crop(tsf_crop_path, enc_preprocessor=None, precision="fp16"):
    if tsf_crop_path is None or tsf_crop_path == "":
        return None
    tsf_np = cv2.imread(tsf_crop_path)
    if tsf_np is None:
        raise FileNotFoundError(f"Could not read TSF crop: {tsf_crop_path}")
    tsf_np = cv2.cvtColor(tsf_np, cv2.COLOR_BGR2RGB)
    tsf_pil = Image.fromarray(tsf_np)
    tsf_tensor = enc_preprocessor.preprocess(tsf_pil)["images"].unsqueeze(0)  # (1, 3, 336, 336)
    tsf_tensor = tsf_tensor.bfloat16() if precision == "bf16" else (tsf_tensor.half() if precision == "fp16" else tsf_tensor.float())
    tsf_tensor = tsf_tensor.cuda(non_blocking=True)
    return [tsf_tensor]


def _read_bbox_txt_xywh(tsf_boxes_txt):
    if tsf_boxes_txt is None or tsf_boxes_txt == "":
        return []
    if not os.path.isfile(tsf_boxes_txt):
        raise FileNotFoundError(f"Could not read TSF bbox txt: {tsf_boxes_txt}")

    boxes = []
    with open(tsf_boxes_txt, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            parts = re.split(r"[\s,]+", line)
            if len(parts) < 4:
                continue
            a, b, c, d = map(float, parts[:4])
            boxes.append((a, b, c, d))
    return boxes


def _infer_bbox_format(raw_boxes, w_img, h_img):
    """
    Infer whether bbox lines are xyxy (x1,y1,x2,y2) or xywh (x,y,w,h).
    Falls back to xywh when ambiguous.
    """
    if not raw_boxes:
        return "xywh"

    votes_xyxy = 0
    votes_xywh = 0
    for a, b, c, d in raw_boxes:
        xyxy_geom = (c > a) and (d > b)
        xywh_geom = (c > 0) and (d > 0)

        if xyxy_geom:
            votes_xyxy += 1
        else:
            votes_xywh += 2

        if xywh_geom:
            votes_xywh += 1

        if xyxy_geom:
            if (0 <= a <= w_img) and (0 <= c <= w_img) and (0 <= b <= h_img) and (0 <= d <= h_img):
                votes_xyxy += 1
            else:
                votes_xywh += 1

        if xywh_geom:
            x2w = a + c
            y2h = b + d
            if (0 <= a <= w_img) and (0 <= x2w <= w_img) and (0 <= b <= h_img) and (0 <= y2h <= h_img):
                votes_xywh += 1
            else:
                votes_xyxy += 1

    return "xyxy" if votes_xyxy > votes_xywh else "xywh"


def _bbox_to_xyxy(raw_box, bbox_format):
    a, b, c, d = raw_box
    if bbox_format == "xyxy":
        return a, b, c, d
    return a, b, a + c, b + d


def preprocess_tsf_from_bbox_txt(
    tsf_boxes_txt,
    np_images,
    input_type,
    conv_generator,
    enc_preprocessor=None,
    precision="fp16",
):
    if tsf_boxes_txt is None or tsf_boxes_txt == "":
        return None
    boxes = _read_bbox_txt_xywh(tsf_boxes_txt)
    if len(boxes) == 0:
        return None

    if input_type == "image":
        frame_indices = [0]
    else:
        frames = np_images[0]
        num_frames = len(frames)
        if num_frames == 0:
            return None
        max_num_frames = conv_generator.NUM_FRAMES
        if num_frames > max_num_frames:
            frame_indices = np.linspace(0, num_frames - 1, max_num_frames, dtype=int).tolist()
        else:
            frame_indices = list(range(num_frames))

    ref_frame = np_images[0][frame_indices[0]]
    ref_h, ref_w = ref_frame.shape[:2]
    bbox_format = _infer_bbox_format(boxes, ref_w, ref_h)

    tsf_tensors = []
    for raw_frame_idx in frame_indices:
        frame = np_images[0][raw_frame_idx]
        h_img, w_img = frame.shape[:2]
        box_idx = min(raw_frame_idx, len(boxes) - 1)
        raw_box = boxes[box_idx]
        x1f, y1f, x2f, y2f = _bbox_to_xyxy(raw_box, bbox_format)
        x1 = int(np.floor(x1f))
        y1 = int(np.floor(y1f))
        x2 = int(np.ceil(x2f))
        y2 = int(np.ceil(y2f))
        x1 = max(0, min(w_img - 1, x1))
        y1 = max(0, min(h_img - 1, y1))
        x2 = max(x1 + 1, min(w_img, x2))
        y2 = max(y1 + 1, min(h_img, y2))
        crop_np = frame[y1:y2, x1:x2]
        if crop_np.size == 0:
            crop_np = frame

        crop_pil = Image.fromarray(crop_np)
        tsf_tensor = enc_preprocessor.preprocess(crop_pil)["images"]  # (3, 336, 336)
        tsf_tensors.append(tsf_tensor)

    if len(tsf_tensors) == 0:
        return None
    tsf_tensor = torch.stack(tsf_tensors, dim=0)  # (N_tsf, 3, 336, 336)
    tsf_tensor = tsf_tensor.bfloat16() if precision == "bf16" else (tsf_tensor.half() if precision == "fp16" else tsf_tensor.float())
    tsf_tensor = tsf_tensor.cuda(non_blocking=True)
    return [tsf_tensor]


def _sam_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().float()
    pixel = tensor * SAM_PIXEL_STD + SAM_PIXEL_MEAN
    pixel = pixel.round().clamp(0, 255).to(torch.uint8)
    array = pixel.permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(array)


def _boxes_to_pixel_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if boxes is None or boxes.numel() == 0:
        return torch.empty((0, 4), dtype=torch.int64)
    boxes = boxes.detach().cpu().float().clone()
    is_normalized = bool(torch.max(boxes) <= 1.5 and torch.min(boxes) >= -0.5)
    if is_normalized:
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * float(width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * float(height)
    boxes = boxes.round()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height - 1)
    return boxes.to(torch.int64)


def _draw_boxes_to_file(
    image_pil: Image.Image,
    boxes: torch.Tensor,
    output_path: str,
    width: int = 2,
    labels: list = None,
):
    image_t = pil_to_tensor(image_pil.convert("RGB"))
    if boxes is None or boxes.numel() == 0:
        to_pil_image(image_t).save(output_path)
        return
    drawn = draw_bounding_boxes(image_t, boxes=boxes, width=width, labels=labels)
    to_pil_image(drawn).save(output_path)


def _flatten_selected_boxes(selected_boxes_by_token):
    all_boxes = []
    labels = []
    for token_idx, token_boxes in enumerate(selected_boxes_by_token or []):
        if token_boxes is None or token_boxes.numel() == 0:
            continue
        token_boxes = token_boxes.detach().cpu()
        all_boxes.append(token_boxes)
        labels.extend([f"tok{token_idx}"] * token_boxes.shape[0])
    if len(all_boxes) == 0:
        return torch.empty((0, 4), dtype=torch.float32), []
    return torch.cat(all_boxes, dim=0), labels


def _write_proposals_txt(path: str, proposals: dict):
    with open(path, "w") as f:
        if proposals is None:
            f.write("No proposals.\n")
            return
        boxes = proposals.get("boxes")
        scores = proposals.get("scores")
        if boxes is None or boxes.numel() == 0:
            f.write("No proposal boxes.\n")
            return
        boxes = boxes.detach().cpu().float()
        scores = scores.detach().cpu().float() if scores is not None else None
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].tolist()
            if scores is not None and i < scores.shape[0]:
                f.write(f"{i}\t{x1:.6f}\t{y1:.6f}\t{x2:.6f}\t{y2:.6f}\t{scores[i].item():.6f}\n")
            else:
                f.write(f"{i}\t{x1:.6f}\t{y1:.6f}\t{x2:.6f}\t{y2:.6f}\n")


def _write_selected_txt(path: str, selected_boxes_by_token):
    with open(path, "w") as f:
        if selected_boxes_by_token is None or len(selected_boxes_by_token) == 0:
            f.write("No selected boxes.\n")
            return
        for token_idx, token_boxes in enumerate(selected_boxes_by_token):
            if token_boxes is None or token_boxes.numel() == 0:
                continue
            token_boxes = token_boxes.detach().cpu().float()
            for box_idx in range(token_boxes.shape[0]):
                x1, y1, x2, y2 = token_boxes[box_idx].tolist()
                f.write(f"tok{token_idx}\t{box_idx}\t{x1:.6f}\t{y1:.6f}\t{x2:.6f}\t{y2:.6f}\n")


def save_proposal_debug(
    proposal_debug: dict,
    images_for_sam_sample: torch.Tensor,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)
    base_img = _sam_tensor_to_pil(images_for_sam_sample)
    img_t = pil_to_tensor(base_img)
    _, h, w = img_t.shape

    _draw_boxes_to_file(base_img, torch.empty((0, 4), dtype=torch.int64), os.path.join(save_dir, "frame_for_proposals.jpg"))

    proposals = proposal_debug.get("proposals") if proposal_debug is not None else None
    selected_before = proposal_debug.get("selected_boxes_before_filter", []) if proposal_debug is not None else []
    selected_after = proposal_debug.get("selected_boxes_after_filter", []) if proposal_debug is not None else []

    prop_boxes_px = _boxes_to_pixel_xyxy(proposals.get("boxes"), w, h) if proposals is not None else torch.empty((0, 4), dtype=torch.int64)
    _draw_boxes_to_file(
        base_img,
        prop_boxes_px,
        os.path.join(save_dir, "proposals_before_filter.jpg"),
        width=2,
    )

    before_boxes_norm, before_labels = _flatten_selected_boxes(selected_before)
    before_boxes_px = _boxes_to_pixel_xyxy(before_boxes_norm, w, h)
    _draw_boxes_to_file(
        base_img,
        before_boxes_px,
        os.path.join(save_dir, "selected_before_filter.jpg"),
        width=3,
        labels=before_labels if len(before_labels) == before_boxes_px.shape[0] else None,
    )

    after_boxes_norm, after_labels = _flatten_selected_boxes(selected_after)
    after_boxes_px = _boxes_to_pixel_xyxy(after_boxes_norm, w, h)
    _draw_boxes_to_file(
        base_img,
        after_boxes_px,
        os.path.join(save_dir, "selected_after_filter.jpg"),
        width=4,
        labels=after_labels if len(after_labels) == after_boxes_px.shape[0] else None,
    )

    _write_proposals_txt(os.path.join(save_dir, "proposals_before_filter.txt"), proposals)
    _write_selected_txt(os.path.join(save_dir, "selected_before_filter.txt"), selected_before)
    _write_selected_txt(os.path.join(save_dir, "selected_after_filter.txt"), selected_after)

    with open(os.path.join(save_dir, "debug_info.txt"), "w") as f:
        if proposal_debug is None:
            f.write("proposal_debug is None\n")
            return
        f.write(f"enabled: {proposal_debug.get('enabled')}\n")
        f.write(f"reason: {proposal_debug.get('reason')}\n")

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    for key in sorted(vars(args).keys()):
        print(f"  {key}: {getattr(args, key)}")

    # Load model, tokenizer, and preprocessors.
    model, tokenizer, enc_preprocessor, conv_generator, sam_preprocessor = initialize_model_videogptplus(
        model_base=args.llava_version_or_path,
        precision=args.precision,
        local_rank=args.local_rank,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        use_sam2_video_branch=args.use_sam2_video_branch,
        base_llm_type=args.base_model_type.split('|')[1],
        use_detection_guidance=args.use_detection_guidance,
        ddetr_model_path=args.ddetr_model_path,
        proposal_topk=args.proposal_topk,
        proposal_score_power=args.proposal_score_power,
        proposal_score_threshold=args.proposal_score_threshold,
        use_proposal_box_regression=args.use_proposal_box_regression,
        proposal_box_loss_weight=args.proposal_box_loss_weight,
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
    runtime = {
        "branch": bool(args.use_sam2_video_branch),
        "model": model,
        "tokenizer": tokenizer,
        "conv_generator": conv_generator,
    }

    def switch_runtime(target_branch: bool):
        if runtime["branch"] == target_branch:
            return
        print(f"Reloading model for use_sam2_video_branch={target_branch} ...")
        old_model = runtime["model"]
        del old_model
        torch.cuda.empty_cache()
        re_model, re_tokenizer, _, re_conv_generator, _ = initialize_model_videogptplus(
            model_base=args.llava_version_or_path,
            precision=args.precision,
            local_rank=args.local_rank,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            use_sam2_video_branch=target_branch,
            base_llm_type=args.base_model_type.split('|')[1],
            use_detection_guidance=args.use_detection_guidance,
            ddetr_model_path=args.ddetr_model_path,
            proposal_topk=args.proposal_topk,
            proposal_score_power=args.proposal_score_power,
            proposal_score_threshold=args.proposal_score_threshold,
            use_proposal_box_regression=args.use_proposal_box_regression,
            proposal_box_loss_weight=args.proposal_box_loss_weight,
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
        runtime["branch"] = target_branch
        runtime["model"] = re_model
        runtime["tokenizer"] = re_tokenizer
        runtime["conv_generator"] = re_conv_generator

    ext = os.path.splitext(args.input_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        input_type = "image"
        np_images = load_image(args.input_path)
        enc_image, enc_context_image, image_sam, original_size_list, resize_list = preprocess_vision(
            np_images,
            type="image",
            enc_preprocessor=enc_preprocessor,
            sam_preprocessor=sam_preprocessor,
            conv_generator=conv_generator,
            precision=args.precision,
        )
    else:
        input_type = "video"
        np_images = load_video(args.input_path)
        enc_image, enc_context_image, image_sam, original_size_list, resize_list = preprocess_vision(
            np_images,
            type="video",
            enc_preprocessor=enc_preprocessor,
            sam_preprocessor=sam_preprocessor,
            conv_generator=conv_generator,
            precision=args.precision,
        )

    tsf_images = None
    if args.tsf_crop_path:
        tsf_images = preprocess_tsf_crop(
            tsf_crop_path=args.tsf_crop_path,
            enc_preprocessor=enc_preprocessor,
            precision=args.precision,
        )
    elif args.tsf_boxes_txt:
        tsf_images = preprocess_tsf_from_bbox_txt(
            tsf_boxes_txt=args.tsf_boxes_txt,
            np_images=np_images,
            input_type=input_type,
            conv_generator=conv_generator,
            enc_preprocessor=enc_preprocessor,
            precision=args.precision,
        )

    save_name = os.path.splitext(os.path.basename(args.input_path))[0]
    save_root = os.path.join(args.vis_save_path, save_name)
    if os.path.exists(save_root):
        print(f"{save_root} exists, deleting it.")
        shutil.rmtree(save_root)
    os.makedirs(save_root, exist_ok=True)
    with open(os.path.join(save_root, "run_args.txt"), "w") as f:
        for key in sorted(vars(args).keys()):
            f.write(f"{key}: {getattr(args, key)}\n")

    if args.proposal_debug_modes == "both":
        modes_to_run = [False]
        if runtime["model"].config.use_sam2:
            modes_to_run = [False, True]
    else:
        modes_to_run = [args.use_sam2_video_branch]
    if input_type != "video":
        modes_to_run = [False]

    autocast_enabled = args.precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    for mode in modes_to_run:
        if mode and input_type != "video":
            print("Skipping sam2_video_branch=True for image input.")
            continue
        if mode and (not runtime["model"].config.use_sam2):
            print("Skipping sam2_video_branch=True because this model is not SAM2.")
            continue

        switch_runtime(mode)
        active_model = runtime["model"]
        active_tokenizer = runtime["tokenizer"]
        active_conv_generator = runtime["conv_generator"]
        if input_type == "image":
            input_ids = active_conv_generator.apply_for_chat(args.prompt_text, type='image', tokenizer=active_tokenizer)
        else:
            input_ids = active_conv_generator.apply_for_chat(args.prompt_text, type='video', tokenizer=active_tokenizer)

        if mode:
            visual_model = getattr(active_model.model, "visual_model", None)
            if not hasattr(visual_model, "init_state_from_tensor"):
                print("Skipping sam2_video_branch=True because loaded SAM2 module does not expose video APIs.")
                continue

        mode_name = "sam2_video_branch_on" if mode else "sam2_video_branch_off"
        mode_save_dir = os.path.join(save_root, mode_name)
        os.makedirs(mode_save_dir, exist_ok=True)
        print(f"Running inference: {mode_name}")

        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
            if args.save_proposal_debug:
                output_ids_batch, video_segments_batch, proposal_debug_batch = active_model.inference(
                    images=enc_image,
                    context_images=enc_context_image,
                    images_for_sam=image_sam,
                    tsf_images=tsf_images,
                    input_ids=input_ids,
                    resize_list=resize_list,
                    original_size_list=original_size_list,
                    max_new_tokens=args.max_new_tokens,
                    use_sam2_video_branch=mode,
                    return_proposal_debug=True,
                )
            else:
                output_ids_batch, video_segments_batch = active_model.inference(
                    images=enc_image,
                    context_images=enc_context_image,
                    images_for_sam=image_sam,
                    tsf_images=tsf_images,
                    input_ids=input_ids,
                    resize_list=resize_list,
                    original_size_list=original_size_list,
                    max_new_tokens=args.max_new_tokens,
                    use_sam2_video_branch=mode,
                )
                proposal_debug_batch = None

        assert len(output_ids_batch) == 1 and len(video_segments_batch) == 1, "Batch size must be 1"
        output_ids = output_ids_batch[0]
        output_ids = output_ids[output_ids != IMAGE_TOKEN_INDEX]
        output_ids = output_ids[output_ids != TSF_TOKEN_INDEX]
        text_output = active_tokenizer.decode(output_ids, skip_special_tokens=False)
        print(f"text_output ({mode_name}): {text_output}")

        batch_idx = 0
        video_frames_np = np_images[batch_idx]
        video_segments = video_segments_batch[batch_idx]
        write_masks(video_segments, video_frames_np, mode_save_dir)

        with open(os.path.join(mode_save_dir, "caption.txt"), "w") as f:
            f.write(text_output)

        if args.save_proposal_debug:
            debug_sample = proposal_debug_batch[batch_idx] if (proposal_debug_batch is not None and len(proposal_debug_batch) > batch_idx) else None
            proposal_dir = os.path.join(mode_save_dir, "proposal_debug")
            save_proposal_debug(
                proposal_debug=debug_sample,
                images_for_sam_sample=image_sam[batch_idx],
                save_dir=proposal_dir,
            )
