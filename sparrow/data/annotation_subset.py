import hashlib
import json
import os
import tempfile
from decimal import Decimal


def _require_ijson():
    try:
        import ijson
        return ijson
    except ImportError as exc:
        raise ImportError(
            "Annotation-level subsampling requires `ijson`. Install it with `pip install ijson`."
        ) from exc


def _json_default(obj):
    if isinstance(obj, Decimal):
        if obj == obj.to_integral_value():
            return int(obj)
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _iter_top_level_object(input_json, prefix):
    ijson = _require_ijson()
    with open(input_json, "rb") as f:
        for obj in ijson.items(f, prefix):
            return obj
    return None


def _image_shard_bucket(image_id, num_shards):
    if isinstance(image_id, int):
        return image_id % num_shards
    return hash(str(image_id)) % num_shards


def _stream_subset(input_json, ratio, seed, images_ndjson, anns_ndjson, num_shards=None, shard_index=0):
    ijson = _require_ijson()
    import random

    rng = random.Random(seed)
    selected_image_ids = set()
    total_images = 0
    kept_images = 0

    with open(input_json, "rb") as f, open(images_ndjson, "w", encoding="utf-8") as out:
        for image in ijson.items(f, "images.item"):
            total_images += 1
            image_id = image.get("id")
            if image_id is None:
                continue
            if num_shards is not None:
                keep = _image_shard_bucket(image_id, num_shards) == shard_index
            else:
                keep = rng.random() <= ratio
            if keep:
                selected_image_ids.add(image_id)
                out.write(json.dumps(image, separators=(",", ":"), default=_json_default))
                out.write("\n")
                kept_images += 1

    total_anns = 0
    kept_anns = 0
    with open(input_json, "rb") as f, open(anns_ndjson, "w", encoding="utf-8") as out:
        for ann in ijson.items(f, "annotations.item"):
            total_anns += 1
            if ann.get("image_id") in selected_image_ids:
                out.write(json.dumps(ann, separators=(",", ":"), default=_json_default))
                out.write("\n")
                kept_anns += 1

    return {
        "total_images": total_images,
        "kept_images": kept_images,
        "total_annotations": total_anns,
        "kept_annotations": kept_anns,
    }


def _write_json_array_from_ndjson(out_f, ndjson_path):
    first = True
    with open(ndjson_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not first:
                out_f.write(",")
            out_f.write(line)
            first = False


def _write_subset_json(input_json, output_json, images_ndjson, anns_ndjson):
    info = _iter_top_level_object(input_json, "info") or {}

    ijson = _require_ijson()
    licenses = []
    with open(input_json, "rb") as f:
        for x in ijson.items(f, "licenses.item"):
            licenses.append(x)

    categories = []
    with open(input_json, "rb") as f:
        for x in ijson.items(f, "categories.item"):
            categories.append(x)

    with open(output_json, "w", encoding="utf-8") as out:
        out.write("{")
        out.write('"info":')
        json.dump(info, out, separators=(",", ":"), default=_json_default)
        out.write(',"licenses":')
        json.dump(licenses, out, separators=(",", ":"), default=_json_default)
        out.write(',"images":[')
        _write_json_array_from_ndjson(out, images_ndjson)
        out.write('],"annotations":[')
        _write_json_array_from_ndjson(out, anns_ndjson)
        out.write('],"categories":')
        json.dump(categories, out, separators=(",", ":"), default=_json_default)
        out.write("}")


def _resolve_ann_path(ann_file):
    return os.path.abspath(os.path.expanduser(ann_file))


def _build_cache_path(ann_file_abs, ratio, seed, dataset_type, cache_dir, num_shards=None, shard_index=0):
    stat = os.stat(ann_file_abs)
    mode = "shard" if num_shards is not None else "ratio"
    if num_shards is not None:
        key = (
            f"{ann_file_abs}|{dataset_type}|{mode}|num_shards={num_shards}|"
            f"shard_index={shard_index}|{stat.st_size}|{stat.st_mtime_ns}"
        )
    else:
        key = f"{ann_file_abs}|{dataset_type}|{mode}|ratio={ratio:.8f}|seed={seed}|{stat.st_size}|{stat.st_mtime_ns}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

    ann_name = os.path.basename(ann_file_abs)
    stem, _ = os.path.splitext(ann_name)
    if num_shards is not None:
        out_name = f"{stem}.{dataset_type}.shard_{shard_index}_of_{num_shards}.{digest}.json"
    else:
        out_name = f"{stem}.{dataset_type}.ratio_{ratio:.4f}.seed_{seed}.{digest}.json"
    return os.path.join(cache_dir, out_name)


def prepare_coco_subset_annotation(
    ann_file,
    ratio,
    dataset_type,
    seed=42,
    cache_dir=None,
    num_shards=None,
    shard_index=0,
):
    """
    Create (or reuse) a cached COCO-format subset annotation file using streaming parse.
    Returns the subset annotation json path.
    """
    if num_shards is None and ratio >= 1:
        return ann_file
    if num_shards is None and ratio <= 0:
        raise ValueError("ratio must be in (0, 1].")
    if num_shards is not None:
        if num_shards <= 1:
            raise ValueError("num_shards must be > 1.")
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError("shard_index must be in [0, num_shards).")

    ann_file_abs = _resolve_ann_path(ann_file)
    if cache_dir is None:
        cache_dir = os.environ.get("GROMA_ANN_CACHE_DIR", "/tmp/groma_ann_subsets")
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    os.makedirs(cache_dir, exist_ok=True)

    output_json = _build_cache_path(
        ann_file_abs, ratio, seed, dataset_type, cache_dir, num_shards=num_shards, shard_index=shard_index
    )
    if os.path.exists(output_json):
        print(f"[dataset-subset] reuse cached subset annotation: {output_json}")
        return output_json

    print(f"[dataset-subset] building subset annotation for {dataset_type}")
    print(f"[dataset-subset] source: {ann_file_abs}")
    if num_shards is not None:
        print(f"[dataset-subset] shard mode: {shard_index}/{num_shards - 1}")
    else:
        print(f"[dataset-subset] ratio mode: ratio={ratio}, seed={seed}")
    print(f"[dataset-subset] cache: {output_json}")

    with tempfile.TemporaryDirectory(prefix="groma_subset_") as td:
        images_ndjson = os.path.join(td, "images.ndjson")
        anns_ndjson = os.path.join(td, "annotations.ndjson")
        stats = _stream_subset(
            ann_file_abs, ratio, seed, images_ndjson, anns_ndjson, num_shards=num_shards, shard_index=shard_index
        )
        _write_subset_json(ann_file_abs, output_json, images_ndjson, anns_ndjson)

    print(
        "[dataset-subset] done: "
        f"images {stats['kept_images']}/{stats['total_images']}, "
        f"annotations {stats['kept_annotations']}/{stats['total_annotations']}"
    )
    return output_json
