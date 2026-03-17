"""Stage-2 training script (detection filter refinement).

This stage trains the proposal filtering heads and LoRA parameters while
keeping heavy visual backbones frozen.
"""

import os
os.environ["MASTER_PORT"] = "29502"  # or 29502, 29510, etc.
os.environ["PYTORCH_NO_SHM"] = "1"  # avoids SharedMemory/posix semaphores
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
import torch
torch.multiprocessing.set_sharing_strategy("file_system")  # avoid /dev/shm


import sys
from functools import partial

import deepspeed
import transformers
from peft import LoraConfig, get_peft_model

from model.SPARROW import SPARROWForCausalLM
from model.videogpt_plus.constants import *

from util.dataset import HybridDataset, collate_fn
from util.trainer import LISATrainer, get_ds_config

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from transformers import AutoTokenizer


@dataclass
class AllArguments:

    # =========================
    # MODEL PATHS / BASE MODEL
    # =========================
    videogptplus_path:str = field(default="checkpoints_hf/MBZUAI/VideoGPT-plus_Phi3-mini-4k/mvbench", metadata={"help": "Path to the base VideoGPT+ checkpoint (kept for compatibility)."})
    videoglamm_path:str = field(default="checkpoints/VideoGPTPlus-Phi3-SAM2-8frame-tunevlproj-epoch29", metadata={"help": "Path to the SPARROW/VideoGLaMM base checkpoint to load."})

    base_llm_path:str = field(default="microsoft/Phi-3-mini-4k-instruct", metadata={"help": "Reference base LLM id/path (kept for compatibility)."})
    base_type:str=field(default="phi3", metadata={"help": "Backbone LLM family. Supported: phi3."}) # phi3|llama3_1
    pretrain_mm_mlp_adapter:str=field(default="", metadata={"help": "Optional path to pretrained video projector weights."})
    pretrain_image_mm_mlp_adapter:str=field(default="", metadata={"help": "Optional path to pretrained image projector weights."})

    # use_mm_start_end:bool=field(default=True)
    

    # =========================
    # MODEL ARCHITECTURE CONFIG
    # =========================
    precision:str=field(default="fp16",metadata={"help": "fp32|fp16|bf16"})
    image_size:int=field(default=1024, metadata={"help": "Input image resolution used by the image branch."})
    model_max_length:int=field(default=2048, metadata={"help": "Maximum sequence length for tokenizer/model context."})
    use_tsf_token:bool=field(default=False, metadata={"help": "Enable optional Target-Specific Features (TSF) token and feature injection."})
    tsf_k:int=field(default=4, metadata={"help": "Number of representative TSF tokens selected via k-means."})
    tsf_kmeans_iters:int=field(default=8, metadata={"help": "K-means iterations for TSF token selection."})

    vision_tower:str=field(default="OpenGVLab/InternVideo2-Stage2_1B-224p-f4/InternVideo2-stage2_1b-224p-f4.pt", metadata={"help": "Video encoder checkpoint path."})
    image_vision_tower:str=field(default="openai/clip-vit-large-patch14-336", metadata={"help": "Image encoder identifier/path."})

    load_in_8bit:bool=field(default=False, metadata={"help": "Load the LLM in 8-bit quantized mode."})
    load_in_4bit:bool=field(default=False, metadata={"help": "Load the LLM in 4-bit quantized mode."})


    # =========================
    # SAM / SEGMENTATION MODULE
    # =========================
    sam_pretrained_path:str=field(default="checkpoints/sam2_hiera_large.pt", metadata={"help": "SAM/SAM2 checkpoint path."})
    out_dim:int=field(default=256, metadata={"help": "Projection dimension from language hidden states into SAM prompt space."})

    tune_mm_mlp_adapter:bool=field(default=False, metadata={"help": "Legacy flag kept for compatibility; base adapters remain frozen in Stage-2."})
    train_mask_decoder:bool=field(default=False, metadata={"help": "Whether to unfreeze SAM mask decoder weights."})

    use_sam_version:str=field(default="v2", metadata={"help": "v1|v1_itm|v2"})
    use_sam2_video_branch:bool=field(default=True, metadata={"help": "Enable SAM2 video-branch model initialization/inference path."})

    num_frames_for_sam:int=field(default=8, metadata={"help": "Number of sampled video frames used by SAM/SAM2."})


    # =========================
    # DATASET CONFIGURATION
    # =========================
    dataset:str=field(default="sem_seg||refer_seg||vqa||reason_seg||grandf||video_vqa||anet_gcg||video_gcg||mevis_gcg||vidstg_gcg||hcstvg_gcg||lasot||got10k_train||got10k_val||mevis_sa2va||rvos||revos||sam2||refer_vos||mevis||vidstg||a2d_sentences||hc_stvg_train||hc_stvg_val||vid_train", metadata={"help": "Datasets joined by '||' for HybridDataset sampling."})
    sample_rates_for_datasets:str=field(default="", metadata={"help": "Comma-separated sampling weights aligned with --dataset order."})

    sem_seg_data:str=field(default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary", metadata={"help": "Semantic segmentation datasets (legacy, pipe-separated)."})
    refer_seg_data:str=field(default="refclef||refcoco||refcoco+||refcocog", metadata={"help": "Referring segmentation datasets (legacy, pipe-separated)."})
    vqa_data:str=field(default="llava_instruct_150k", metadata={"help": "VQA datasets (legacy, pipe-separated)."})
    reason_seg_data:str=field(default="ReasonSeg|train", metadata={"help": "Reasoning segmentation datasets (legacy, pipe-separated)."})
    reason_seg_explanatory:float=field(default=0.1, metadata={"help": "Sampling ratio for explanatory ReasonSeg examples."})

    refer_vos_data:str=field(default="ytvos||davis17", metadata={"help": "Referring video object segmentation datasets."})
    video_vqa_data:str=field(default="video_instruct_100k", metadata={"help": "Video QA datasets."})
    video_tg_data:str=field(default="charades||anetcaps||qvh", metadata={"help": "Temporal grounding datasets."})

    val_dataset:str=field(default="ReasonSeg|val", metadata={"help": "Validation dataset spec (reserved for compatibility)."})

    dataset_dir:str=field(default="datasets/image", metadata={"help": "Root directory for image datasets."})
    video_dataset_dir:str=field(default="datasets/video", metadata={"help": "Root directory for video datasets."})
    sa2va_video_data_root:str=field(
        default="",
        metadata={"help": "Optional root for SA2VA-style video data (contains revos/ rvos/ mevis/ sam_v_full)."},
    )
    vot_data_root:str=field(
        default="",
        metadata={"help": "Optional root for VOT datasets (contains GOT-10k/ LaSOT/ A2D_sentences/ HC-STVG/ VID-Sentence)."},
    )
    visual_token_reserve:int=field(
        default=255,
        metadata={"help": "Reserved visual-token budget used by util/dataset.py truncation logic."},
    )

    num_classes_per_sample:int=field(default=3, metadata={"help": "Maximum number of target classes sampled per training item."})


    # =========================
    # TRAINING CONFIGURATION
    # =========================
    local_rank:int=field(default=0, metadata={"help": "Local CUDA rank/device index."})

    epochs:int=field(default=8, metadata={"help": "Number of training epochs."})
    steps_per_epoch:int=field(default=500, metadata={"help": "Optimizer steps per epoch."})

    batch_size:int=field(default=1, metadata={"help": "Per-GPU micro-batch size."})
    grad_accumulation_steps:int=field(default=1, metadata={"help": "Gradient accumulation steps."})
    val_batch_size:int=field(default=1, metadata={"help": "Validation batch size (reserved for compatibility)."})

    workers:int=field(default=0, metadata={"help": "Data loader worker processes per local rank."})
    lazy_load_heavy_datasets:bool=field(
        default=True,
        metadata={"help": "Lazily initialize heavy datasets (sam2/revos/rvos/val_*)."},
    )
    max_loaded_heavy_datasets:int=field(
        default=1,
        metadata={"help": "Maximum number of heavy datasets kept in RAM at once."},
    )
    dataset_sticky_steps:int=field(
        default=32,
        metadata={"help": "Reuse sampled dataset for N steps before re-sampling."},
    )

    lr:float=field(default=0.0003, metadata={"help": "Base learning rate."})

    beta1:float=field(default=0.9, metadata={"help": "AdamW beta1."})
    beta2:float=field(default=0.95, metadata={"help": "AdamW beta2."})

    ce_loss_weight:float=field(default=1.0, metadata={"help": "Weight for language cross-entropy alignment loss."})
    dice_loss_weight:float=field(default=0.5, metadata={"help": "Weight for Dice mask loss."})
    bce_loss_weight:float=field(default=2.0, metadata={"help": "Weight for BCE mask loss."})

    gradient_checkpointing:bool=field(default=True, metadata={"help": "Enable gradient checkpointing for memory savings."})


    # =========================
    # LORA CONFIGURATION
    # =========================
    lora_r:int=field(default=8, metadata={"help": "LoRA rank."})
    lora_alpha:int=field(default=16, metadata={"help": "LoRA scaling alpha."})
    lora_dropout:float=field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_target_modules:str=field(default="q_proj,v_proj", metadata={"help": "Comma-separated linear module name fragments to LoRA-wrap in the LLM."})


    # =========================
    # DETECTION GUIDANCE (DDETR)
    # =========================
    use_detection_guidance:bool=field(default=True, metadata={"help": "Enable DDETR proposal guidance."})
    ddetr_model_path:str=field(default="checkpoints_hf/ddetr_sam2", metadata={"help": "Path to pretrained CustomDDETRSAM2Model checkpoint."})

    proposal_loss_weight:float=field(default=1.0, metadata={"help": "Weight for detection-guided selection loss."})
    proposal_iou_threshold:float=field(default=0.3, metadata={"help": "IoU threshold to treat proposals as positives."})
    proposal_topk:int=field(default=100, metadata={"help": "Number of top proposals kept from DDETR."})

    proposal_selector_hidden_dim:int=field(default=256, metadata={"help": "Hidden dimension for proposal selector head."})
    proposal_score_threshold:float=field(default=0.4, metadata={"help": "Probability threshold for selecting proposals per SEG token."})
    max_selected_boxes_per_token:int=field(default=100, metadata={"help": "Maximum selected proposals per SEG token after thresholding."})
    use_proposal_box_regression:bool=field(default=False, metadata={"help": "Enable proposal box regression branch (L1 + GIoU) on positive matches."})
    proposal_box_loss_weight:float=field(default=0.01, metadata={"help": "Weight for proposal box regression loss; keep small for minimal effect."})
    use_roi_helper:bool=field(default=True, metadata={"help": "Enable additive ROI helper head for proposal selection."})
    roi_helper_hidden_dim:int=field(default=256, metadata={"help": "Hidden dimension for ROI helper head."})
    roi_align_output_size:int=field(default=7, metadata={"help": "ROIAlign output size for ROI helper head."})
    roi_fusion_init:float=field(default=0.1, metadata={"help": "Initial fusion weight for ROI helper logits."})
    use_box_prior_filter:bool=field(default=False, metadata={"help": "Filter selected guidance boxes using SAM2 box-to-mask priors."})
    box_prior_min_score:float=field(default=0.35, metadata={"help": "Minimum prior quality score for keeping a selected box."})
    box_prior_max_keep:int=field(default=4, metadata={"help": "Maximum kept boxes per SEG token after prior filtering."})


    # =========================
    # EVALUATION
    # =========================
    eval_only:bool=field(default=False, metadata={"help": "Run in evaluation-only mode."})
    eval_num_frames:int=field(default=-1, metadata={"help": "Optional frame limit for evaluation routines."})


    # =========================
    # CHECKPOINT / RESUME
    # =========================
    resume_dir:str=field(default="", metadata={"help": "Checkpoint directory to resume from."})
    auto_resume:bool=field(default=True, metadata={"help": "Automatically resume from latest checkpoint in run directory if present."})
    start_epoch:int=field(default=0, metadata={"help": "Manual epoch offset (reserved for compatibility)."})

    intermediate_weight:str=field(default="", metadata={"help": "Path to intermediate weights when exporting HF model."})


    # =========================
    # LOGGING / OUTPUT
    # =========================
    print_freq:int=field(default=1, metadata={"help": "Logging frequency in training steps."})
    vis_save_path:str = field(default="vis_output", metadata={"help": "Visualization output directory (reserved for compatibility)."})

    exp_name:str=field(default='train', metadata={"help": "Experiment name used in log/checkpoint subdirectories."})

    logs_base_dir:str=field(default='runs/logs/', metadata={"help": "Base directory for TensorBoard logs."})
    ckpt_base_dir:str=field(default='runs/ckpts/', metadata={"help": "Base directory for checkpoints."})

    save_hf_model:bool=field(default=False, metadata={"help": "Export merged HF model and exit."})
    hf_save_path:str=field(default="hf_model", metadata={"help": "Output path for exported HF model."})


def main():

    parser = HfArgumentParser(AllArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Bridge simple training args to util/dataset.py environment-based path resolution.
    if args.sa2va_video_data_root:
        os.environ["SA2VA_VIDEO_DATA_ROOT"] = args.sa2va_video_data_root
    if args.vot_data_root:
        os.environ["VOT_DATA_ROOT"] = args.vot_data_root
    os.environ["SPARROW_VISUAL_TOKEN_RESERVE"] = str(args.visual_token_reserve)

    def build_model_args():
        model_args = {
            "train_mask_decoder": args.train_mask_decoder,
            "out_dim": args.out_dim,
            "mask_decoder_itm": (True if args.use_sam_version == "v1_itm" else False),
            "use_sam2": (True if args.use_sam_version == "v2" else False),
            "sam_pretrained_path": args.sam_pretrained_path,
            "use_detection_guidance": args.use_detection_guidance,
            "ddetr_model_path": args.ddetr_model_path if args.ddetr_model_path else None,
            "proposal_loss_weight": args.proposal_loss_weight,
            "proposal_iou_threshold": args.proposal_iou_threshold,
            "proposal_topk": args.proposal_topk,
            "proposal_selector_hidden_dim": args.proposal_selector_hidden_dim,
            "proposal_score_threshold": args.proposal_score_threshold,
            "max_selected_boxes_per_token": args.max_selected_boxes_per_token,
            "use_proposal_box_regression": args.use_proposal_box_regression,
            "proposal_box_loss_weight": args.proposal_box_loss_weight,
            "use_roi_helper": args.use_roi_helper,
            "roi_helper_hidden_dim": args.roi_helper_hidden_dim,
            "roi_align_output_size": args.roi_align_output_size,
            "roi_fusion_init": args.roi_fusion_init,
            "use_box_prior_filter": args.use_box_prior_filter,
            "box_prior_min_score": args.box_prior_min_score,
            "box_prior_max_keep": args.box_prior_max_keep,
        }
        if not (args.eval_only or args.save_hf_model):
            model_args.update(
                {
                    "ce_loss_weight": args.ce_loss_weight,
                    "dice_loss_weight": args.dice_loss_weight,
                    "bce_loss_weight": args.bce_loss_weight,
                }
            )
        return model_args

    torch_dtype = torch.bfloat16 if args.precision == "bf16" else (torch.half if args.precision == "fp16" else torch.float32)


    def load_videoglamm_model_from_base(
            model_base,
            load_in_8bit=False,
            load_in_4bit=False,
            base_type="phi3",
        ):

        model_args = build_model_args()
        
        if load_in_4bit:
            model_args.update(
                {
                    "torch_dtype": torch.float16,
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
                    "torch_dtype": torch.float16,
                    "quantization_config": transformers.BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_skip_modules=["visual_model"],
                    ),
                }
            )
        

        # -----------------------
        # Load model
        # -----------------------
        print("Loading SPARROW from base model...")
        if base_type == "phi3":
            model = SPARROWForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=False,
                use_sam2_video_branch=False,
                **model_args,
            )

        else:
            raise ValueError("Invalid base_type. Should be either phi3 or llama3_1")

        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

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

        # Sync token ids into config
        model.config.seg_token_idx = seg_token_idx
        model.config.box_token_idx = box_token_idx
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        # llama3_1 fallback
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048

        print("Model is loaded...")

        return model, tokenizer, context_len, seg_token_idx, mm_use_im_start_end, mm_use_im_patch_token

    if args.base_type=="phi3":
        model, tokenizer, context_len, seg_token_idx, use_mm_start_end, _mm_use_im_patch_token = load_videoglamm_model_from_base(args.videoglamm_path, load_in_8bit=False, load_in_4bit=False, base_type=args.base_type)
    else:
        raise ValueError("Invalid base_type")

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_tsf_token = args.use_tsf_token
    model.config.tsf_k = args.tsf_k
    model.config.tsf_kmeans_iters = args.tsf_kmeans_iters

    # If training, enable gradient checkpointing
    if not (args.eval_only or args.save_hf_model):
        model.enable_input_require_grads()
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()


    # conversation_generator
    from util.conv_generator import ConvGenerator_VideoGPTPlus
    conversation_generator = ConvGenerator_VideoGPTPlus(
        use_mm_start_end=use_mm_start_end,
        base_type=args.base_type,
        use_tsf_token=args.use_tsf_token,
    )

    # enc_preprocessor
    from util.enc_preprocessors import EncPreprocessor_VideoGPTPlus
    enc_preprocessor = EncPreprocessor_VideoGPTPlus()

    ### SAM preprocessor
    if args.use_sam_version=="v1" or args.use_sam_version=="v1_itm":
        from util.sam_transforms import SAM_v1_Preprocess
        sam_preprocessor = SAM_v1_Preprocess()
    elif args.use_sam_version=="v2":
        from util.sam_transforms import SAM_v2_Preprocess
        sam_preprocessor = SAM_v2_Preprocess()
        
    ### Initialize encoder modules
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    image_vision_tower = model.get_model().get_image_vision_tower()
    image_vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    if not (args.eval_only or args.save_hf_model):
        # Freeze visual encoders.
        for p in vision_tower.parameters():
            p.requires_grad = False
        for p in image_vision_tower.parameters():
            p.requires_grad = False

        # Keep base adapters and proposal modules frozen.
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False
        for p in model.get_model().image_mm_projector.parameters():
            p.requires_grad = False
        for p in model.get_model().text_hidden_fcs.parameters():
            p.requires_grad = False

        if hasattr(model.get_model(), "ddetr_helper") and model.get_model().ddetr_helper is not None:
            for p in model.get_model().ddetr_helper.model.parameters():
                p.requires_grad = False
        if hasattr(model.get_model(), "proposal_selector") and model.get_model().proposal_selector is not None:
            for p in model.get_model().proposal_selector.parameters():
                p.requires_grad = False
        if hasattr(model.get_model(), "roi_helper") and model.get_model().roi_helper is not None:
            for p in model.get_model().roi_helper.parameters():
                p.requires_grad = False

        for p in model.parameters():
            p.requires_grad = False

    ### Setup LoRA settings
    # ================================================================================================================================================
    def _collect_linear_fullnames(model: torch.nn.Module, roots: list[str]) -> list[str]:
        wanted = set()
        for full_name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                for r in roots:
                    # match only layers strictly under the desired root
                    if full_name.startswith(r + ".") or full_name == r:
                        wanted.add(full_name)
                        break
        return sorted(wanted)

    def get_model_with_lora(
        model,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
        # what to target
        include_adapters: bool = True,
        include_text_hidden_fcs: bool = True,
        include_transformer: bool = True,
        transformer_target_modules: list[str] | None = None,  # e.g. ["q_proj", "v_proj"]
        # guard-rails / matching behavior
        transformer_exclude_prefixes: tuple[str, ...] = (
            "visual_model",           # grounding encoder-decoder
            "vision_tower", "image_vision_tower",
            "mm_projector", "image_mm_projector",
            "text_hidden_fcs",
            "ddetr_helper"
        ),
        verbose: bool = True,
    ):
        """
        Inject a single LoRA config that targets:
        - Adapters (mm_projector/image_mm_projector/box_mm_projector) when include_adapters=True
        - text_hidden_fcs when include_text_hidden_fcs=True
        - Transformer attention MLP/attn projections (e.g., q_proj/v_proj) when include_transformer=True

        This merges your two previous utilities so you only wrap the model once.
        """

        if transformer_target_modules is None:
            transformer_target_modules = ["q_proj", "v_proj"]

        # --- Gather adapter linear layers (projectors, etc.) ---
        adapter_targets = set()
        if include_adapters or include_text_hidden_fcs:
            # many HF wrappers keep the actual module under .get_model()
            core = model.get_model() if hasattr(model, "get_model") else model

            target_roots = []

            if hasattr(core, "mm_projector") and core.mm_projector is not None:
                if include_adapters:
                    target_roots.append("model.mm_projector")

            if hasattr(core, "image_mm_projector") and core.image_mm_projector is not None:
                if include_adapters:
                    target_roots.append("model.image_mm_projector")

            if hasattr(core, "text_hidden_fcs") and core.text_hidden_fcs is not None:
                if include_text_hidden_fcs:
                    target_roots.append("model.text_hidden_fcs")

            # some repos register without the "model." prefix—cover both
            alt_roots = [r.replace("model.", "", 1) for r in target_roots]

            for roots in (target_roots, alt_roots):
                for name in _collect_linear_fullnames(model, roots):
                    adapter_targets.add(name)

        # --- Gather transformer linear layers (q_proj/v_proj by name) ---
        transformer_targets = set()
        if include_transformer:
            for name, module in model.named_modules():
                if not isinstance(module, torch.nn.Linear):
                    continue
                # skip known non-transformer regions
                if any(name.startswith(pref) or f".{pref}." in name for pref in transformer_exclude_prefixes):
                    continue
                # include if name contains any desired submodule token (e.g. q_proj, v_proj)
                if any(tok in name for tok in transformer_target_modules):
                    transformer_targets.add(name)

        # --- Union of targets ---
        all_targets = sorted(adapter_targets | transformer_targets)

        if not all_targets:
            raise RuntimeError(
                "LoRA merge: No nn.Linear targets found. "
                "Check name prefixes or disable one of include_adapters/include_text_hidden_fcs/include_transformer."
            )

        # --- Configure & wrap once ---
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=all_targets,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # --- Diagnostics ---
        if verbose:
            print("==== LoRA injection report ====")
            if include_adapters or include_text_hidden_fcs:
                print(f"Adapter targets matched: {len(adapter_targets)}")
                for n in sorted(adapter_targets):
                    print("  [adapter] ", n)
            if include_transformer:
                print(f"Transformer targets matched: {len(transformer_targets)}")
                for n in sorted(transformer_targets):
                    print("  [xform]   ", n)
            print(f"TOTAL targets: {len(all_targets)}")

            # What actually got lora_A/B after PEFT wrap (sanity check)
            injected = 0
            for n, m in model.named_modules():
                if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
                    injected += 1
            print(f"PEFT modules with LoRA params: {injected}")

            model.print_trainable_parameters()

        return model

    # ================================================================================================================================================
    lora_r = args.lora_r
    if lora_r > 0:
        transformer_targets = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        model = get_model_with_lora(
            model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            include_adapters=False,
            include_text_hidden_fcs=True,
            include_transformer=True,
            transformer_target_modules=transformer_targets if len(transformer_targets) > 0 else ["q_proj", "v_proj"],
        )

    if args.save_hf_model:

        state_dict = torch.load(args.intermediate_weight, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

        model = model.merge_and_unload()
        state_dict = {}
        for k, v in model.state_dict().items():
            if "vision_tower" not in k and "image_vision_tower" not in k:
                state_dict[k] = v
                
        model.generation_config.do_sample = True
        model.config.architectures = ["SPARROWForCausalLM"]
        
        model.save_pretrained(args.hf_save_path, state_dict=state_dict, safe_serialization=False)
        tokenizer.save_pretrained(args.hf_save_path)
        print("Saved model in Huggingface format at: ", args.hf_save_path)
        sys.exit(0)
    

    # --- Re-enable proposal heads ---
    assert hasattr(model.get_model(), "proposal_selector") and model.get_model().proposal_selector is not None, \
        "proposal_selector not found on model.get_model()."

    for p in model.get_model().proposal_selector.parameters():
        p.requires_grad = True

    if hasattr(model.get_model(), "roi_helper") and model.get_model().roi_helper is not None:
        for p in model.get_model().roi_helper.parameters():
            p.requires_grad = True
    if (
        hasattr(model.get_model(), "roi_helper")
        and model.get_model().roi_helper is not None
        and hasattr(model.get_model(), "roi_fusion_gamma")
    ):
        model.get_model().roi_fusion_gamma.requires_grad = True

    # text_hidden_fcs base weights stay frozen; LoRA params on this block remain trainable.

    if lora_r > 0:
        model.print_trainable_parameters()

    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_params)


    # ================================================
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    # ================================================

    ### Load train and validation datasets
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    dataset_names_list = args.dataset.split("||")
    if args.sample_rates_for_datasets=="": #default
        sample_rates = [1.0] * len(dataset_names_list)
    else:
        sample_rates = [float(x) for x in args.sample_rates_for_datasets.split(",")]
    assert len(dataset_names_list) == len(sample_rates), "Number of datasets and sample rates should be equal."
    num_samples_per_epoch=args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size

    # Train dataset
    train_dataset = HybridDataset(
        base_image_dir = args.dataset_dir,
        base_video_dir = args.video_dataset_dir,

        enc_preprocessor = enc_preprocessor,
        sam_preprocessor = sam_preprocessor,
        conversation_generator = conversation_generator,

        num_samples_per_epoch=num_samples_per_epoch,
        num_classes_per_sample=args.num_classes_per_sample,
        
        dataset=dataset_names_list,
        sample_rate=sample_rates,
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
        refer_vos_data=args.refer_vos_data,
        video_vqa_data=args.video_vqa_data,
        video_tg_data=args.video_tg_data,
        
        num_frames_for_sam=args.num_frames_for_sam,
        reason_seg_explanatory=args.reason_seg_explanatory,
        lazy_load_heavy_datasets=args.lazy_load_heavy_datasets,
        max_loaded_heavy_datasets=args.max_loaded_heavy_datasets,
        dataset_sticky_steps=args.dataset_sticky_steps,
    )

    data_collator = partial(
        collate_fn,
        tokenizer=tokenizer,
        local_rank=args.local_rank,
        conversation_generator=conversation_generator
    )

    # DeepSpeed
    print("Initializing DeepSpeed")
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=data_collator,
        config=get_ds_config(args),
    )

    # DeepSpeed defaults num_local_io_workers to 2*device_count unless overridden.
    # Rebuild loader explicitly so args.workers controls multiprocessing.
    train_loader = model_engine.deepspeed_io(
        train_dataset,
        collate_fn=data_collator,
        num_local_io_workers=args.workers,
        pin_memory=False,
    )


    # Trainer
    print("Initializing Trainer")
    trainer = LISATrainer(model_engine, train_loader, scheduler, args, exp_name=args.exp_name)


    # Resume training
    if args.auto_resume:
        if os.path.exists(os.path.join(trainer.ckpt_save_dir, "latest")):
            trainer.resume(trainer.ckpt_save_dir)

    elif args.resume_dir:
        trainer.resume(args.resume_dir)


    # Start training
    print("Starting training")
    trainer.train(args.epochs)



if __name__ == '__main__':
    main()
