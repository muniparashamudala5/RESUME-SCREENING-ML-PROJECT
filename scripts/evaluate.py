"""Measure the screener against human decisions.

    python scripts/evaluate.py samples samples/labels.csv

labels.csv has one row per (resume, role) pair:  filename,role,label   (label: shortlist | reject)

The shipped samples/labels.csv is a toy file that only proves the script runs. Tune thresholds on
real resumes labelled by recruiters (aim for 100+), and keep a separate set for the final check.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,  # noqa: E402
                             recall_score, roc_auc_score)

from app.config import get_settings  # noqa: E402
from app.extractor import ExtractionError  # noqa: E402
from app.scorer import Screener, UnknownRoleError  # noqa: E402


def report(title: str, y_true: list[int], y_pred: list[int]) -> None:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    print(f"\n{title}")
    print(f"  accuracy {accuracy_score(y_true, y_pred):.2f}   "
          f"precision {precision_score(y_true, y_pred, zero_division=0):.2f}   "
          f"recall {recall_score(y_true, y_pred, zero_division=0):.2f}   "
          f"F1 {f1_score(y_true, y_pred, zero_division=0):.2f}")
    print(f"  confusion: true shortlist caught {tp}, missed {fn}; true reject wrongly passed {fp}, correctly rejected {tn}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder containing the PDF resumes")
    parser.add_argument("labels", type=Path, help="CSV with columns filename,role,label")
    args = parser.parse_args()

    screener = Screener.from_settings(get_settings())
    y_true, scores, decisions, rows = [], [], [], []
    with open(args.labels, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                result = screener.screen_pdf((args.folder / row["filename"]).read_bytes(), row["role"])
            except (ExtractionError, FileNotFoundError, UnknownRoleError) as exc:
                print(f"skipped {row['filename']} ({row['role']}): {exc}")
                continue
            y_true.append(1 if row["label"].strip().lower() == "shortlist" else 0)
            scores.append(result["score"])
            decisions.append(result["decision"])
            rows.append(row)

    if not y_true:
        sys.exit("No resumes could be evaluated.")
    print(f"Evaluated {len(y_true)} resume/role pairs ({sum(y_true)} human shortlists).")

    report("Strict: only 'Shortlisted' counts as a pass", y_true, [int(d == "Shortlisted") for d in decisions])
    report("Lenient: 'Shortlisted' and 'Needs review' both pass", y_true, [int(d != "Not shortlisted") for d in decisions])

    if len(set(y_true)) == 2:
        print(f"\nROC AUC of the raw score: {roc_auc_score(y_true, scores):.2f}")
        print("\nScore threshold sweep (pass if score >= threshold)")
        print("  threshold  precision  recall     F1")
        for threshold in range(30, 95, 5):
            pred = [int(s >= threshold) for s in scores]
            print(f"  {threshold:9d}  {precision_score(y_true, pred, zero_division=0):9.2f}  "
                  f"{recall_score(y_true, pred, zero_division=0):6.2f}  {f1_score(y_true, pred, zero_division=0):5.2f}")

    strict = [int(d == "Shortlisted") for d in decisions]
    wrong = [(r, s, d) for r, s, d, t, p in zip(rows, scores, decisions, y_true, strict) if t != p]
    if wrong:
        print("\nDisagreements with the human label (read these first when tuning):")
        for row, score, decision in wrong:
            print(f"  {row['filename']} / {row['role']}: human={row['label']}, screener={decision} ({score})")


if __name__ == "__main__":
    main()
