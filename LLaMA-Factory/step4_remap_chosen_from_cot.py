"""Remap chosen SQL in the 500-sample file to CoT outputs.

This script matches each sample's `chosen` SQL against the SQL block in
`mydata/dpo_datasets_cot_sft.json` and replaces `chosen` with the full
`output` (reasoning + SQL) from the CoT file. A new JSON file is produced
without mutating the source files.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Tuple

SOURCE_PATH = "mydata/dpo_datasets_500_sample_207.json"
CoT_PATH = "mydata/dpo_datasets_cot_sft.json"
OUTPUT_PATH = "mydata/dpo_datasets_500_sample_207_cot.json"


def extract_sql_block(text: str) -> str:
    """Return the first ```sql ... ``` block content; fallback to empty string."""
    m = re.search(r"```sql\s*(.*?)\s*```", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    return ""


def normalize_sql(sql: str) -> str:
    """Whitespace-insensitive, case-insensitive canonical form for matching."""
    cleaned = sql.strip().rstrip(";")
    # Collapse whitespace and lowercase for lenient matching
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower()


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_sql_to_output_map(cot_items: List[Dict]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in cot_items:
        output = item.get("output", "")
        sql = extract_sql_block(output)
        key = normalize_sql(sql) if sql else ""
        if not key:
            continue
        # Only keep the first occurrence to avoid accidental overrides
        mapping.setdefault(key, output)
    return mapping


def remap(samples: List[Dict], sql_map: Dict[str, str]) -> Tuple[List[Dict], int, int]:
    updated: List[Dict] = []
    hit = 0
    miss = 0

    for item in samples:
        chosen_sql = item.get("chosen", "")
        norm = normalize_sql(chosen_sql)
        replacement = sql_map.get(norm)
        if replacement:
            hit += 1
            new_chosen = replacement
        else:
            miss += 1
            new_chosen = chosen_sql

        updated.append(
            {
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "chosen": new_chosen,
                "rejected": item.get("rejected", ""),
            }
        )

    return updated, hit, miss


def main() -> None:
    print(f"Loading source: {SOURCE_PATH}")
    samples = load_json(SOURCE_PATH)

    print(f"Loading CoT outputs: {CoT_PATH}")
    cot_items = load_json(CoT_PATH)

    sql_map = build_sql_to_output_map(cot_items)
    print(f"Built SQL->output map: {len(sql_map)} entries")

    updated, hit, miss = remap(samples, sql_map)
    print(f"Matched: {hit}, Unmatched: {miss}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)
    print(f"Written new file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()