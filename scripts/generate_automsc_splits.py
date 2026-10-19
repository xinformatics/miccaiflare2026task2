#!/usr/bin/env python3
"""Generate deterministic classification-stratified AutoMSC folds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedKFold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("preprocessed", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    with (args.dataset / "cls_data.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    identifier_column = (
        "identifier" if "identifier" in rows[0] else "case_id" if "case_id" in rows[0] else next(iter(rows[0]))
    )
    identifiers = [row[identifier_column] for row in rows]
    dataset_json = json.loads((args.dataset / "dataset.json").read_text())
    declared_heads = list(dataset_json.get("classification_labels", {}))
    if "label" in rows[0]:
        label_columns = ["label"]
    else:
        label_columns = [column for column in declared_heads if column in rows[0]]
        if not label_columns:
            label_columns = [
                column for column in rows[0] if column != identifier_column and all(row[column].isdigit() for row in rows)
            ]
    if not label_columns:
        raise ValueError(f"Cannot infer classification columns from {list(rows[0])}")
    joint_labels = ["|".join(row[column] for column in label_columns) for row in rows]
    # If a multi-head combination is too rare, stratify on the first head rather than failing.
    counts = Counter(joint_labels)
    labels = joint_labels if min(counts.values()) >= args.folds else [row[label_columns[0]] for row in rows]
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    splits = []
    for train, validation in splitter.split(identifiers, labels):
        splits.append(
            {
                "train": [identifiers[index] for index in train],
                "val": [identifiers[index] for index in validation],
            }
        )
    output = args.preprocessed / "splits_final.json"
    output.write_text(json.dumps(splits, indent=2) + "\n")
    print(output)
    for index, split in enumerate(splits):
        counts: dict[str, int] = {}
        validation_set = set(split["val"])
        for identifier, label in zip(identifiers, labels):
            if identifier in validation_set:
                counts[label] = counts.get(label, 0) + 1
        print(f"fold {index}: train={len(split['train'])}, val={len(split['val'])}, val_classes={counts}")


if __name__ == "__main__":
    main()
