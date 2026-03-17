import json
import os
from tqdm import tqdm
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Preprocess HC-STVG-v2 annotations.")
    parser.add_argument(
        "--base_video_dataset_dir",
        default=os.getenv("VIDEO_DATASET_DIR", "datasets/video"),
        help="Root video dataset directory that contains hcstvg/ and processed/.",
    )
    args = parser.parse_args()

    base_video_dataset_dir = args.base_video_dataset_dir

    video_path = os.path.join(base_video_dataset_dir, "hcstvg", "Video")
    ann_path = os.path.join(base_video_dataset_dir, "hcstvg", "anno_v2")

    processed_ann_path = os.path.join(base_video_dataset_dir, "processed/hcstvg/hcstvg_annotations")
    os.makedirs(processed_ann_path, exist_ok=True)

    # get video to path mapping
    dirs = os.listdir(video_path)
    vid2path = {}
    for dir in dirs:
        files = os.listdir(os.path.join(video_path, dir))
        for file in files:
            assert os.path.exists(os.path.join(video_path, dir, file))
            vid2path[file[:-4]] = os.path.join(dir, file)

    # preproc annotations
    files = ["train_v2.json", "val_v2.json"]
    for file in files:
        videos = []
        with open(os.path.join(ann_path, file), "r") as f:
            annotations = json.load(f)
        for video, annot in tqdm(annotations.items()):
            out = {
                "original_video_id": video[:-4],
                "frame_count": annot["img_num"],
                "width": annot["img_size"][1],
                "height": annot["img_size"][0],
                "tube_start_frame": annot["st_frame"],  # starts with 1
                "tube_end_frame": annot["st_frame"] + len(annot["bbox"]),  # excluded
                "tube_start_time": annot["st_time"],
                "tube_end_time": annot["ed_time"],
                "video_path": vid2path[video[:-4]],
                "caption": annot["English"],
                "video_id": len(videos),
                "trajectory": annot["bbox"],
            }
            videos.append(out)

        with open(os.path.join(processed_ann_path, file[:-5] + "_proc.json"), "w") as f:
            json.dump(videos, f)
