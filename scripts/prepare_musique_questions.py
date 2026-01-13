#!/usr/bin/env python3
"""Prepare MuSiQue question JSONL files under data/raw/questions.

Downloads MuSiQue from the Hugging Face Hub (default: dgslibisey/MuSiQue) and
writes files compatible with PRMRAG batch scripts:

  data/raw/questions/musique_train.jsonl
  data/raw/questions/musique_validation.jsonl

Schema (per line):
  {
    "_id": str,
    "question": str,
    "answer": str,
    "answer_aliases": list[str],
    "type": "musique",
    "level": str | null,              # e.g., "2hop", "3hop1", "4hop3"
    "supporting_titles": list[str],   # from MuSiQue paragraphs (is_supporting=True)
    "question_decomposition": list[dict],  # optional via --include-decomposition
    "paragraphs": list[dict],              # optional via --include-paragraphs
    "source_dataset": str,                 # HF dataset id
    "source_split": str,                   # split name
  }
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable


def _infer_level(example_id: str) -> str | None:
    if not example_id:
        return None
    m = re.match(r"^([^_]+)__", example_id)
    return m.group(1) if m else None


def _supporting_titles(paragraphs: Iterable[Dict[str, Any]]) -> list[str]:
    titles: list[str] = []
    for p in paragraphs:
        if p.get("is_supporting") and p.get("title"):
            titles.append(p["title"])
    # Preserve order but de-duplicate
    seen: set[str] = set()
    deduped: list[str] = []
    for t in titles:
        if t in seen:
            continue
        seen.add(t)
        deduped.append(t)
    return deduped


def write_split(
    *,
    dataset,
    split: str,
    out_path: Path,
    source_dataset: str,
    limit: int | None,
    include_paragraphs: bool,
    include_decomposition: bool,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(dataset[split]):
            if limit is not None and i >= limit:
                break

            ex_id = ex.get("id")
            paragraphs = ex.get("paragraphs") or []

            record: Dict[str, Any] = {
                "_id": ex_id,
                "question": ex.get("question", ""),
                "answer": ex.get("answer", ""),
                "answer_aliases": ex.get("answer_aliases") or [],
                "type": "musique",
                "level": _infer_level(ex_id),
                "supporting_titles": _supporting_titles(paragraphs),
                "source_dataset": source_dataset,
                "source_split": split,
            }

            if include_decomposition:
                record["question_decomposition"] = ex.get("question_decomposition") or []
            if include_paragraphs:
                record["paragraphs"] = paragraphs

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

    return num_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/convert MuSiQue to PRMRAG JSONL format")
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="dgslibisey/MuSiQue",
        help="Hugging Face dataset id to load (default: dgslibisey/MuSiQue)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/questions"),
        help="Output directory (default: data/raw/questions)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "validation"],
        help="Splits to export (default: train validation)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-split limit (for testing)",
    )
    parser.add_argument(
        "--include-paragraphs",
        action="store_true",
        help="Include full MuSiQue paragraphs (large)",
    )
    parser.add_argument(
        "--include-decomposition",
        action="store_true",
        help="Include question decomposition fields",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(args.hf_dataset)
    available_splits = set(dataset.keys())

    for split in args.splits:
        if split not in available_splits:
            raise SystemExit(f"Split '{split}' not found in dataset. Available: {sorted(available_splits)}")

        out_path = args.output_dir / f"musique_{split}.jsonl"
        if out_path.exists() and not args.force:
            raise SystemExit(f"Refusing to overwrite existing file: {out_path} (use --force)")

        n = write_split(
            dataset=dataset,
            split=split,
            out_path=out_path,
            source_dataset=args.hf_dataset,
            limit=args.limit,
            include_paragraphs=args.include_paragraphs,
            include_decomposition=args.include_decomposition,
        )
        print(f"✓ Wrote {n} examples to {out_path}")


if __name__ == "__main__":
    main()

