#!/usr/bin/env python3
"""
Filter a COCO-format annotation file by image file_name prefix (streaming).
Useful when only a subset of shard folders exists locally (e.g. SA1B sa_000000..sa_000050).

Requires:
  pip install ijson
"""

import argparse
import json
import os
import tempfile
from decimal import Decimal


def _require_ijson():
    try:
        import ijson
        return ijson
    except ImportError as exc:
        raise ImportError("Missing dependency: ijson. Install it with `pip install ijson`.") from exc


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


def _allowed_prefixes(start_idx, end_idx, folder_pattern):
    return tuple(folder_pattern.format(i) + "/" for i in range(start_idx, end_idx + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--prefix-start", type=int, required=True)
    parser.add_argument("--prefix-end", type=int, required=True)
    parser.add_argument("--folder-pattern", type=str, default="sa_{:06d}")
    args = parser.parse_args()

    if args.prefix_start > args.prefix_end:
        raise ValueError("--prefix-start must be <= --prefix-end")

    ijson = _require_ijson()
    allowed = _allowed_prefixes(args.prefix_start, args.prefix_end, args.folder_pattern)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="coco_filter_prefix_") as td:
        images_ndjson = os.path.join(td, "images.ndjson")
        anns_ndjson = os.path.join(td, "annotations.ndjson")

        selected_image_ids = set()
        total_images = 0
        kept_images = 0

        print(f"Filtering images by prefixes: {allowed[0]} ... {allowed[-1]}", flush=True)
        with open(args.input_json, "rb") as f, open(images_ndjson, "w", encoding="utf-8") as out:
            for image in ijson.items(f, "images.item"):
                total_images += 1
                fname = image.get("file_name", "")
                if fname.startswith(allowed):
                    image_id = image.get("id")
                    if image_id is None:
                        continue
                    selected_image_ids.add(image_id)
                    out.write(json.dumps(image, separators=(",", ":"), default=_json_default))
                    out.write("\n")
                    kept_images += 1
                if total_images % 200000 == 0:
                    print(f"[images] processed={total_images:,}, kept={kept_images:,}", flush=True)

        total_anns = 0
        kept_anns = 0
        print("Filtering annotations...", flush=True)
        with open(args.input_json, "rb") as f, open(anns_ndjson, "w", encoding="utf-8") as out:
            for ann in ijson.items(f, "annotations.item"):
                total_anns += 1
                if ann.get("image_id") in selected_image_ids:
                    out.write(json.dumps(ann, separators=(",", ":"), default=_json_default))
                    out.write("\n")
                    kept_anns += 1
                if total_anns % 1000000 == 0:
                    print(f"[annotations] processed={total_anns:,}, kept={kept_anns:,}", flush=True)

        info = _iter_top_level_object(args.input_json, "info") or {}

        licenses = []
        with open(args.input_json, "rb") as f:
            for x in ijson.items(f, "licenses.item"):
                licenses.append(x)

        categories = []
        with open(args.input_json, "rb") as f:
            for x in ijson.items(f, "categories.item"):
                categories.append(x)

        print(f"Writing output: {args.output_json}", flush=True)
        with open(args.output_json, "w", encoding="utf-8") as out:
            out.write("{")
            out.write('"info":')
            json.dump(info, out, separators=(",", ":"), default=_json_default)
            out.write(',"licenses":')
            json.dump(licenses, out, separators=(",", ":"), default=_json_default)
            out.write(',"images":[')
            _write_array_from_ndjson(out, images_ndjson)
            out.write('],"annotations":[')
            _write_array_from_ndjson(out, anns_ndjson)
            out.write('],"categories":')
            json.dump(categories, out, separators=(",", ":"), default=_json_default)
            out.write("}")

        print(
            "Done. "
            f"images {kept_images:,}/{total_images:,}, "
            f"annotations {kept_anns:,}/{total_anns:,}",
            flush=True,
        )


if __name__ == "__main__":
    main()
