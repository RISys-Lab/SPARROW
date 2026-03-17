# Dataset

This project is trained on a merged collection of datasets assembled from two sources:

- **VideoGLaMM dataset specification**  
- **Sa2VA-Training dataset release**

The goal is to unify image-grounded, referring segmentation, reasoning segmentation, dense captioning, and video understanding data into a single training layout.

## Sources

### 1) VideoGLaMM
The [VideoGLaMM Dataset](https://github.com/mbzuai-oryx/VideoGLaMM/blob/main/Dataset.md) specification includes:
- LISA datasets
- GranDf
- image datasets extracted under `./dataset`
- video datasets extracted under `./video_dataset`

VideoGLaMM documents the expected top-level structure as:

```text
dataset/
├── ade20k
├── coco
├── cocostuff
├── grandf_dataset
├── llava_dataset
├── mapillary
├── other
├── reason_seg
├── refer_seg
└── vlpart

video_dataset/
├── activitynet
├── activitynet_captions
├── activitynet_entities
├── activitynet_entities_gcg
├── burst
├── hcstvg
├── hcstvg_gcg
├── mevis
├── mevis_gcg
├── processed
├── refer_davis
├── refer_youtube_vos
├── video_gcg
├── video_instruct_100k
├── vidstg
├── vidstg_gcg
├── ytvis
└── ytvos_gcg
```

---

### 2) Sa2VA
The download link is [here](https://huggingface.co/datasets/Dense-World/Sa2VA-Training).

Please directly put the zip files into the `root` directory and unzip them. For example, you can download the `video_datas_mevis.zip` and unzip it in the `root` directory like:
```bash
unzip video_datas_mevis.zip
```

The final data structure should be like:
```
data/
├── video_datas
|   ├── revos
|   ├── mevis
|   └── davis17
|   └── chat_univi
|   └── sam_v_full # [!important] please download this from sam-2 directly.
|   └── Ref-SAV.json
```
**Important**: `sam_v_full` is the SA-V dataset, which is not included in the download link. You can download it from **Meta** ([here](https://ai.meta.com/datasets/segment-anything-video/)). Please follow their license.
</details>

---

### 3) TSF Dataset
Please refer to **soon**
