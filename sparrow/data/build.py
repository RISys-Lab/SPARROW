import copy
import os
import torch
import numpy as np
from mmcv import Config
from torch.utils.data import ConcatDataset
from sparrow.data.annotation_subset import prepare_coco_subset_annotation

from sparrow.data.datasets.coco import COCODet
from sparrow.data.datasets.refcoco_rec import RefCOCO
from sparrow.data.datasets.flickr import Flickr30k
from sparrow.data.datasets.grit import Grit
from sparrow.data.datasets.refcoco_cap import RefCOCOCap
from sparrow.data.datasets.llava import LLaVAInstruct
from sparrow.data.datasets.groma import GromaInstruct
from sparrow.data.datasets.visual_genome import SingleRoundVG, MultiRoundsVG
from sparrow.data.datasets.det_data import ClassAgnosticCoCo, ClassAgnosticSA1B


def build_multi_datasets(dataset_cfg_file, tokenizer=None, **kwargs):
    dataset_cfgs = Config.fromfile(dataset_cfg_file)
    dataset_cfgs = dataset_cfgs.datasets
    assert isinstance(dataset_cfgs, list)
    datasets = [build_dataset(cfg, tokenizer=tokenizer, **kwargs) for cfg in dataset_cfgs]
    return ConcatDataset(datasets)


def build_dataset(dataset_cfg, tokenizer=None, **kwargs):
    dataset_type = dataset_cfg.pop('type')
    ratio = dataset_cfg.pop('ratio', 1)
    subset_seed = dataset_cfg.pop('subset_seed', 42)
    subset_num_shards = dataset_cfg.pop('subset_num_shards', None)
    subset_shard_index = dataset_cfg.pop('subset_shard_index', None)
    subset_non_overlap = dataset_cfg.pop('subset_non_overlap', False)
    conv_temp = dataset_cfg.pop('conv_temp', 'default')

    # For detection pretrain datasets, apply ratio at annotation-file level first
    # so we avoid loading the full source JSON into RAM.
    if dataset_type in ('coco_box', 'obj365_box', 'openimage_box', 'v3det_box', 'sa1b_box') and ratio < 1:
        if 'ann_file' not in dataset_cfg:
            raise KeyError(f"dataset {dataset_type} must provide 'ann_file'.")

        if subset_non_overlap and subset_num_shards is None:
            inv_ratio = 1.0 / ratio
            rounded = int(round(inv_ratio))
            if abs(inv_ratio - rounded) > 1e-6:
                raise ValueError(
                    f"subset_non_overlap requires ratio to be reciprocal of an integer, got ratio={ratio}."
                )
            subset_num_shards = rounded

        if subset_num_shards is not None and subset_shard_index is None:
            subset_shard_index = int(os.environ.get('GROMA_SUBSET_SHARD_INDEX', 0))

        dataset_cfg['ann_file'] = prepare_coco_subset_annotation(
            ann_file=dataset_cfg['ann_file'],
            ratio=ratio,
            dataset_type=dataset_type,
            seed=subset_seed,
            num_shards=subset_num_shards,
            shard_index=subset_shard_index if subset_shard_index is not None else 0,
        )
        ratio = 1

    if dataset_type in ('coco_box', 'obj365_box', 'openimage_box', 'v3det_box'):
        dataset = ClassAgnosticCoCo(**dataset_cfg)
    elif dataset_type == 'sa1b_box':
        dataset = ClassAgnosticSA1B(**dataset_cfg)
    elif dataset_type == 'coco':
        dataset = COCODet(**dataset_cfg, tokenizer=tokenizer, conv_temp=conv_temp)
    elif dataset_type == 'flickr30k':
        dataset = Flickr30k(**dataset_cfg, tokenizer=tokenizer, conv_temp=conv_temp)
    elif dataset_type == 'single_vg':
        dataset = SingleRoundVG(**dataset_cfg, tokenizer=tokenizer, conv_temp=conv_temp)
    elif dataset_type == 'multi_vg':
        dataset = MultiRoundsVG(**dataset_cfg, tokenizer=tokenizer, conv_temp=conv_temp)
    elif dataset_type == 'refcoco_cap':
        dataset = RefCOCOCap(**dataset_cfg, tokenizer=tokenizer, conv_temp=conv_temp)
    elif dataset_type == 'refcoco_rec':
        dataset = RefCOCO(**dataset_cfg, tokenizer=tokenizer, conv_temp=conv_temp)
    elif dataset_type == 'grit':
        dataset = Grit(**dataset_cfg, tokenizer=tokenizer, img_processor=kwargs['img_processor'], conv_temp=conv_temp)
    elif dataset_type == 'llava_instruct':
        dataset = LLaVAInstruct(**dataset_cfg, tokenizer=tokenizer, img_processor=kwargs['img_processor'], conv_temp=conv_temp)
    elif dataset_type == 'groma_instruct':
        dataset = GromaInstruct(**dataset_cfg, tokenizer=tokenizer, img_processor=kwargs['img_processor'], conv_temp=conv_temp)
    else:
        raise NotImplementedError

    if ratio < 1:
        print(f'randomly sample {ratio} of the dataset {dataset_type}: {int(ratio * len(dataset))}')
        random_indices = np.random.choice(len(dataset), int(ratio * len(dataset)), replace=False)
        subsample_dataset = torch.utils.data.Subset(dataset, random_indices)
        return subsample_dataset

    return dataset


if __name__ == '__main__':
    # for quick test
    dataset_cfg_file = 'sparrow/data/configs/vl_finetune.py'
    train_datasets = build_multi_datasets(dataset_cfg_file, tokenizer=None, img_processor=None)
    print(len(train_datasets))
    train_datasets[0]
    import random
    for i in range(10):
        ind = random.randint(0, len(train_datasets))
        train_datasets[ind]


