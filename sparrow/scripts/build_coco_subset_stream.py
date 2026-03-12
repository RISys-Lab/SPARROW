#!/usr/bin/env python3
"""
Build a COCO-format subset JSON with low peak RAM by streaming.

Requires:
  pip install ijson
"""

import argparse
import json
import os
import random
import tempfile
from decimal import Decimal


def _require_ijson():
    try:
        import ijson  # noqa: F401
        return ijson
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: ijson. Install it with `pip install ijson`."
        ) from exc


def _dump_ndjson_line(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, separators=(",", ":"), default=_json_default))
        f.write("\n")


def _json_default(obj):
    if isinstance(obj, Decimal):
        # Keep integer-valued Decimals as integers for id fields.
        if obj == obj.to_integral_value():
            return int(obj)
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _image_shard_bucket(image_id, num_shards):
    if isinstance(image_id, int):
        return image_id % num_shards
    return hash(str(image_id)) % num_shards


def _stream_images(input_json, keep_ratio, seed, tmp_images_path, num_shards, shard_index):
    ijson = _require_ijson()
    rng = random.Random(seed)
    selected_ids = set()
    total_images = 0
    kept_images = 0

    with open(input_json, "rb") as f:
        for image in ijson.items(f, "images.item"):
            total_images += 1
            keep = False
            image_id = image.get("id")
            if num_shards is not None:
                if image_id is not None and _image_shard_bucket(image_id, num_shards) == shard_index:
                    keep = True
            else:
                if rng.random() <= keep_ratio:
                    keep = True

            if keep:
                kept_images += 1
                if image_id is not None:
                    selected_ids.add(image_id)
                    _dump_ndjson_line(tmp_images_path, image)
            if total_images % 200000 == 0:
                print(
                    f"[images] processed={total_images:,}, kept={kept_images:,}",
                    flush=True,
                )

    return selected_ids, total_images, kept_images


def _stream_annotations(input_json, selected_ids, tmp_anns_path):
    ijson = _require_ijson()
    total_anns = 0
    kept_anns = 0

    with open(input_json, "rb") as f:
        for ann in ijson.items(f, "annotations.item"):
            total_anns += 1
            if ann.get("image_id") in selected_ids:
                kept_anns += 1
                _dump_ndjson_line(tmp_anns_path, ann)
            if total_anns % 1000000 == 0:
                print(
                    f"[annotations] processed={total_anns:,}, kept={kept_anns:,}",
                    flush=True,
                )

    return total_anns, kept_anns


def _read_optional_top_level(input_json):
    ijson = _require_ijson()
    info = {}
    licenses = []
    categories = []

    with open(input_json, "rb") as f:
        for obj in ijson.items(f, "info"):
            info = obj or {}
            break

    with open(input_json, "rb") as f:
        for lic in ijson.items(f, "licenses.item"):
            licenses.append(lic)

    with open(input_json, "rb") as f:
        for cat in ijson.items(f, "categories.item"):
            categories.append(cat)

    return info, licenses, categories


def _write_array_from_ndjson(out_f, ndjson_path):
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


def _write_output(output_json, info, licenses, categories, tmp_images_path, tmp_anns_path):
    with open(output_json, "w", encoding="utf-8") as out:
        out.write("{")
        out.write('"info":')
        json.dump(info, out, separators=(",", ":"), default=_json_default)
        out.write(',"licenses":')
        json.dump(licenses, out, separators=(",", ":"), default=_json_default)
        out.write(',"images":[')
        _write_array_from_ndjson(out, tmp_images_path)
        out.write('],"annotations":[')
        _write_array_from_ndjson(out, tmp_anns_path)
        out.write('],"categories":')
        json.dump(categories, out, separators=(",", ":"), default=_json_default)
        out.write("}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--keep-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    if args.num_shards is None and not (0.0 < args.keep_ratio <= 1.0):
        raise ValueError("--keep-ratio must be in (0, 1].")
    if args.num_shards is not None:
        if args.num_shards <= 1:
            raise ValueError("--num-shards must be > 1.")
        if args.shard_index < 0 or args.shard_index >= args.num_shards:
            raise ValueError("--shard-index must be in [0, num_shards).")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="coco_subset_") as td:
        tmp_images_path = os.path.join(td, "images.ndjson")
        tmp_anns_path = os.path.join(td, "annotations.ndjson")

        if args.num_shards is not None:
            print(
                f"Streaming images for shard {args.shard_index}/{args.num_shards - 1}...",
                flush=True,
            )
        else:
            print(f"Streaming images with keep-ratio={args.keep_ratio}...", flush=True)
        selected_ids, total_images, kept_images = _stream_images(
            args.input_json,
            args.keep_ratio,
            args.seed,
            tmp_images_path,
            args.num_shards,
            args.shard_index,
        )
        print(
            f"Images done: processed={total_images:,}, kept={kept_images:,}",
            flush=True,
        )

        print("Streaming annotations...", flush=True)
        total_anns, kept_anns = _stream_annotations(
            args.input_json, selected_ids, tmp_anns_path
        )
        print(
            f"Annotations done: processed={total_anns:,}, kept={kept_anns:,}",
            flush=True,
        )

        print("Reading top-level metadata (info/licenses/categories)...", flush=True)
        info, licenses, categories = _read_optional_top_level(args.input_json)

        print(f"Writing subset JSON: {args.output_json}", flush=True)
        _write_output(
            args.output_json,
            info,
            licenses,
            categories,
            tmp_images_path,
            tmp_anns_path,
        )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
