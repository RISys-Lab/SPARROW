import argparse
import json
import os
import re

import cv2
import numpy as np
import skimage
import torch
from tqdm import tqdm

from util.dataset import ValGCGDataset


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    union_sum = np.sum(union)
    if union_sum == 0:
        return 1.0
    return float(np.sum(intersection) / union_sum)


def compute_miou(pred_masks, gt_masks):
    pred_masks = list(pred_masks)
    gt_masks = list(gt_masks)
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return 0.0

    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)

    paired_iou = []
    while iou_matrix.size > 0 and np.max(iou_matrix) > 0:
        max_iou_idx = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
        paired_iou.append(float(iou_matrix[max_iou_idx]))
        iou_matrix = np.delete(iou_matrix, max_iou_idx[0], axis=0)
        iou_matrix = np.delete(iou_matrix, max_iou_idx[1], axis=1)

    return float(np.mean(paired_iou)) if paired_iou else 0.0


def compute_iou_matrix(pred_masks, gt_masks):
    pred_masks = list(pred_masks)
    gt_masks = list(gt_masks)
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)
    return iou_matrix


def _load_text_similarity_backend():
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import AutoModel, AutoTokenizer

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    bert_model.eval()
    return bert_tokenizer, bert_model, cosine_similarity


def _text_similarity_bert(str1, str2, bert_tokenizer, bert_model, cosine_similarity):
    with torch.no_grad():
        inputs1 = bert_tokenizer(str1, return_tensors="pt", max_length=512, truncation=True)
        emb1 = torch.mean(bert_model(**inputs1).last_hidden_state[0], dim=0).cpu().numpy()
        inputs2 = bert_tokenizer(str2, return_tensors="pt", max_length=512, truncation=True)
        emb2 = torch.mean(bert_model(**inputs2).last_hidden_state[0], dim=0).cpu().numpy()
    return float(cosine_similarity([emb1], [emb2])[0, 0])


def find_best_matches(
    gt_masks,
    gt_labels,
    pred_masks,
    pred_labels,
    iou_threshold=0.5,
    text_sim_threshold=0.5,
    text_similarity_fn=None,
):
    gt_masks = list(gt_masks)
    pred_masks = list(pred_masks)
    ious = compute_iou_matrix(gt_masks, pred_masks)

    text_sims = np.zeros((len(gt_labels), len(pred_labels)), dtype=np.float32)
    for i, gt_label in enumerate(gt_labels):
        for j, pred_label in enumerate(pred_labels):
            if text_similarity_fn is None:
                text_sims[i, j] = 1.0
            else:
                text_sims[i, j] = text_similarity_fn(gt_label, pred_label)

    best_matches = []
    while ious.size > 0:
        max_iou_idx = np.unravel_index(np.argmax(ious), ious.shape)
        if ious[max_iou_idx] < iou_threshold or text_sims[max_iou_idx] < text_sim_threshold:
            break
        best_matches.append(max_iou_idx)
        ious[max_iou_idx[0], :] = 0
        ious[:, max_iou_idx[1]] = 0
        text_sims[max_iou_idx[0], :] = 0
        text_sims[:, max_iou_idx[1]] = 0

    return best_matches


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GCG inference outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video_dataset_dir",
        default="/media/ansari/BFEE-795F/VideoGLaMM/video_dataset/",
        type=str,
        help="Dataset root.",
    )
    parser.add_argument("--vis_save_path", default="vis_output/eval_gcg_new", type=str, help="Root path containing inference outputs.")
    parser.add_argument("--dataset_name", default="video_gcg", type=str, choices=["video_gcg"], help="Evaluated dataset.")
    parser.add_argument("--eval_miou", action="store_true", default=False, help="Evaluate mask mIoU.")
    parser.add_argument("--eval_recall", action="store_true", default=False, help="Evaluate phrase-mask recall with text matching.")
    parser.add_argument("--eval_caption", action="store_true", default=False, help="Evaluate caption quality metrics.")
    parser.add_argument("--use_clair", action="store_true", default=False, help="Use CLAIR for caption evaluation.")
    return parser.parse_args()


def _load_available_results(args, dataset_len):
    all_res = []
    for idx in tqdm(range(dataset_len), desc="Loading saved outputs"):
        try:
            save_dir_for_current_video = os.path.join(args.vis_save_path, args.dataset_name, f"{idx:06d}")
            saved_file = os.path.join(save_dir_for_current_video, "res.json")
            if os.path.exists(saved_file):
                with open(saved_file, "r") as file:
                    all_res.append(json.load(file))
            else:
                all_res.append(None)
        except Exception as e:
            print(f"Error in processing {idx}: {e}")
            all_res.append(None)
    return all_res


def _list_mask_dirs(base_dir, prefix):
    if not os.path.isdir(base_dir):
        return []
    out = []
    for name in os.listdir(base_dir):
        if name.startswith(prefix):
            m = re.match(rf"^{re.escape(prefix)}(.+)$", name)
            if m:
                out.append((m.group(1), os.path.join(base_dir, name)))
    out.sort(key=lambda x: x[0])
    return out


def eval_caption_quality(all_gt_references, all_pred_captions):
    references = {}
    captions = {}
    for k, (gt_ref, pred_caption) in enumerate(
        tqdm(zip(all_gt_references, all_pred_captions), total=len(all_gt_references), desc="Preparing captions")
    ):
        references[str(k)] = [gt_ref[:2000]]
        captions[str(k)] = [pred_caption[:2000]]

    new_cap = [{"image_id": k, "caption": v[0]} for k, v in captions.items()]
    new_ref = {"images": [], "annotations": []}
    for k, refs in references.items():
        new_ref["images"].append({"id": k})
        for ref in refs:
            new_ref["annotations"].append({"image_id": k, "id": k, "caption": ref})

    with open("tmp_references.json", "w") as fgts:
        json.dump(new_ref, fgts)
    with open("tmp_captions.json", "w") as fres:
        json.dump(new_cap, fres)

    from pycocoevalcap.eval import COCOEvalCap
    from pycocotools.coco import COCO

    coco = COCO("tmp_references.json")
    coco_result = coco.loadRes("tmp_captions.json")
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()
    for metric, score in coco_eval.eval.items():
        print(f"\033[92m{metric}: {score:.3f}\033[0m")


def eval_caption_quality_with_clair(all_gt_references, all_pred_captions):
    from util.clair import clair

    sum_score = 0.0
    count = 0
    for gt_ref, pred_caption in tqdm(
        zip(all_gt_references, all_pred_captions),
        total=len(all_gt_references),
        desc="Evaluating CLAIR",
    ):
        clair_score, _ = clair([pred_caption[:2000]], [gt_ref[:2000]], model="chat-gpt")
        sum_score += clair_score
        count += 1
    avg_score = sum_score / max(count, 1)
    print(f"\033[92mCLAIR Score: {avg_score:.3f}\033[0m")


def main():
    args = parse_args()
    if not any([args.eval_miou, args.eval_recall, args.eval_caption, args.use_clair]):
        print("No evaluation flag provided. Use one or more of --eval_miou, --eval_recall, --eval_caption, --use_clair.")
        return

    eval_dataset = ValGCGDataset(args.video_dataset_dir, val_datasets="video_gcg||mevis_gcg||vidstg_gcg")
    print("eval_dataset", len(eval_dataset))
    all_res = _load_available_results(args, dataset_len=len(eval_dataset))
    print("all_res", len(all_res))

    mious = [] if args.eval_miou else None
    iou_threshold = 0.5
    text_sim_threshold = 0.5
    true_positives = 0
    actual_positives = 0
    all_gt_references = []
    all_pred_captions = []

    text_similarity_fn = None
    if args.eval_recall:
        bert_tokenizer, bert_model, cosine_similarity = _load_text_similarity_backend()

        def _sim(a, b):
            return _text_similarity_bert(a, b, bert_tokenizer, bert_model, cosine_similarity)

        text_similarity_fn = _sim

    for idx in tqdm(range(len(all_res)), desc="Computing metrics"):
        res = all_res[idx]
        if res is None:
            if args.eval_miou:
                mious.append(0.0)
            if args.eval_recall:
                actual_positives += 0
            if args.eval_caption or args.use_clair:
                all_gt_references.append("")
                all_pred_captions.append("")
            continue

        try:
            gt_text_cleaned = res["gt_text_cleaned"]
            pred_text_cleaned = res["pred_text_cleaned"]
            gt_phrases = res["gt_phrases"]
            pred_phrases = res["pred_phrases"]

            save_dir_for_current_video = os.path.join(args.vis_save_path, args.dataset_name, f"{idx:06d}")
            img_frames_dir = os.path.join(save_dir_for_current_video, "img_frames")

            if args.eval_miou or args.eval_recall:
                filenames = os.listdir(img_frames_dir)
                sorted_filenames = sorted(filenames, key=lambda x: int(re.search(r"\d+", x).group()))
                images = []
                for filename in sorted_filenames:
                    if filename.endswith(".jpg"):
                        image_path = os.path.join(img_frames_dir, filename)
                        image = cv2.imread(image_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        images.append(image)

                gt_masks = {}
                for obj_id, obj_dir in _list_mask_dirs(save_dir_for_current_video, "gt_masks_"):
                    gt_masks[obj_id] = []
                    for ti in range(len(images)):
                        mask_path = os.path.join(obj_dir, f"mask_{ti}.png")
                        mask_img = skimage.io.imread(mask_path)
                        gt_masks[obj_id].append(mask_img)

                pred_masks = {}
                for obj_id, obj_dir in _list_mask_dirs(save_dir_for_current_video, "pred_masks_"):
                    pred_masks[obj_id] = []
                    for ti in range(len(images)):
                        mask_path = os.path.join(obj_dir, f"mask_{ti}.png")
                        if os.path.exists(mask_path):
                            mask_img = skimage.io.imread(mask_path)
                        else:
                            mask_img = np.zeros(images[ti].shape[:2], dtype=np.uint8)
                        pred_masks[obj_id].append(mask_img)

            if args.eval_miou:
                mious.append(compute_miou(pred_masks.values(), gt_masks.values()))

            if args.eval_recall:
                actual_positives += len(gt_phrases)
                best_matches = find_best_matches(
                    gt_masks.values(),
                    gt_phrases,
                    pred_masks.values(),
                    pred_phrases,
                    iou_threshold=iou_threshold,
                    text_sim_threshold=text_sim_threshold,
                    text_similarity_fn=text_similarity_fn,
                )
                true_positives += len(best_matches)

            if args.eval_caption or args.use_clair:
                all_gt_references.append(gt_text_cleaned)
                all_pred_captions.append(pred_text_cleaned)

        except Exception as e:
            print(f"Error in processing {idx}: {e}")
            if args.eval_miou:
                mious.append(0.0)
            if args.eval_recall:
                actual_positives += len(res.get("gt_phrases", []))
            if args.eval_caption or args.use_clair:
                all_gt_references.append(res.get("gt_text_cleaned", ""))
                all_pred_captions.append("")

    if args.eval_miou:
        mean_miou = float(np.mean(mious)) if mious else 0.0
        print(f"\033[92mMean IoU (mIoU) across all videos: {mean_miou}\033[0m")

    if args.eval_recall:
        recall = true_positives / actual_positives if actual_positives > 0 else 0.0
        print(f"\033[92mRecall: {recall:.3f}\033[0m")

    if args.eval_caption and not args.use_clair:
        print("Evaluating caption quality...")
        eval_caption_quality(all_gt_references, all_pred_captions)

    if args.use_clair:
        print("Evaluating caption quality with CLAIR...")
        eval_caption_quality_with_clair(all_gt_references, all_pred_captions)


if __name__ == "__main__":
    main()
