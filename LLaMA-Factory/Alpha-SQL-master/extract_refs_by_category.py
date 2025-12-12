#!/usr/bin/env python3
"""Extract 【参考信息】 text per category and aggregate to JSON.

Reads a dataset JSON (list of records with `category` and `input`),
parses the text after the marker `【参考信息】` in each `input`, and
writes a JSON file mapping category -> list of unique reference blocks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

REFERENCE_MARK = "【参考信息】"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_reference(text: str) -> str:
    if not text:
        return ""
    if REFERENCE_MARK not in text:
        return ""
    # The marker appears twice (intro sentence and the actual block); keep only the last block.
    ref_body = text.rsplit(REFERENCE_MARK, 1)[-1]
    ref_body = ref_body.lstrip("：:").strip()
    return ref_body


def aggregate_by_category(records: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    bucket: Dict[str, List[str]] = {}
    for rec in records:
        cat = rec.get("category")
        if not cat:
            continue
        ref = parse_reference(rec.get("input", ""))
        if ref == "":
            continue
        # ensure unique per category while preserving insertion order
        arr = bucket.setdefault(cat, [])
        if ref not in arr:
            arr.append(ref)
    return bucket


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract reference text by category")
    parser.add_argument("--input", required=True, help="Path to dataset JSON list")
    parser.add_argument("--output", required=True, help="Path to write aggregated JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        data = load_json(input_path)
    except Exception as exc:
        print(f"Failed to load input: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print("Input JSON must be a list of records", file=sys.stderr)
        sys.exit(1)

    aggregated = aggregate_by_category(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)

    print(f"Categories: {len(aggregated)}")
    total_refs = sum(len(v) for v in aggregated.values())
    print(f"Total reference blocks: {total_refs}")


if __name__ == "__main__":
    main()
