"""
Create a balanced manual-annotation candidate set from data/processed/sentences.csv.

Design:
- Exactly 50 sentences per report when every report has >= 50 sentences.
- 50 reports -> 2,500 candidates.
- Deterministic random seed.
- Within each report, sample across sentence-length deciles to avoid selecting
  only very short or very long sentences.
- Keeps source provenance columns unchanged.
- Writes an audit summary so the sample is reproducible.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math
import pandas as pd


DEFAULT_INPUT = Path("data/processed/sentences.csv")
DEFAULT_OUTPUT = Path("data/annotations/annotation_candidates.csv")
DEFAULT_AUDIT = Path("data/annotations/annotation_sampling_audit.csv")

REQUIRED_COLUMNS = [
    "report_id",
    "page",
    "paragraph_id",
    "sentence_id",
    "text",
]


def stratified_sample(group: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n rows from one report across sentence-length deciles."""
    group = group.copy()
    group["_char_len"] = group["text"].astype(str).str.len()

    # Rank first so qcut behaves well when many sentences have the same length.
    group["_rank"] = group["_char_len"].rank(method="first")
    group["_bin"] = pd.qcut(
        group["_rank"],
        q=min(10, len(group)),
        labels=False,
        duplicates="drop",
    )

    bins = list(group.groupby("_bin", sort=True, observed=True))
    selected_parts = []

    # Allocate nearly equally across length bins.
    base = n // len(bins)
    remainder = n % len(bins)

    rng_seed = seed + sum(ord(c) for c in str(group["report_id"].iloc[0]))

    for i, (_, bucket) in enumerate(bins):
        take = base + (1 if i < remainder else 0)
        take = min(take, len(bucket))

        selected_parts.append(
            bucket.sample(n=take, random_state=rng_seed + i)
        )

    selected = pd.concat(selected_parts, ignore_index=True)

    # Defensive fill if duplicate-length/bin edge cases leave us short.
    if len(selected) < n:
        remaining = group.loc[~group["sentence_id"].isin(selected["sentence_id"])]
        extra = remaining.sample(
            n=n - len(selected),
            random_state=rng_seed + 1000,
        )
        selected = pd.concat([selected, extra], ignore_index=True)

    return selected.drop(columns=["_char_len", "_rank", "_bin"], errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--per-report", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    audit_path = Path(args.audit)

    df = pd.read_csv(input_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["report_id"] = df["report_id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    if df["sentence_id"].duplicated().any():
        raise ValueError("Duplicate sentence_id values detected.")

    report_ids = sorted(df["report_id"].unique())
    if len(report_ids) != 50:
        raise ValueError(
            f"Expected 50 reports, found {len(report_ids)}: {report_ids}"
        )

    too_small = (
        df.groupby("report_id")
        .size()
        .loc[lambda s: s < args.per_report]
    )

    if not too_small.empty:
        raise ValueError(
            "These reports contain fewer than "
            f"{args.per_report} sentences:\n{too_small.to_string()}"
        )

    selected_parts = []
    audit_rows = []

    for report_id in report_ids:
        group = df[df["report_id"] == report_id]
        selected = stratified_sample(
            group,
            n=args.per_report,
            seed=args.seed,
        )

        selected_parts.append(selected)

        audit_rows.append(
            {
                "report_id": report_id,
                "available_sentences": len(group),
                "selected_sentences": len(selected),
                "sampling_method": "10-bin sentence-length stratified random sampling",
                "seed": args.seed,
            }
        )

    candidates = pd.concat(selected_parts, ignore_index=True)

    # Stable ordering for reproducibility and easier annotation.
    candidates = candidates.sort_values(
        ["report_id", "page", "sentence_id"]
    ).reset_index(drop=True)

    # Add an annotation row ID without changing source sentence IDs.
    candidates.insert(
        0,
        "candidate_id",
        [f"CAND_{i:05d}" for i in range(1, len(candidates) + 1)],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    candidates.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(audit_rows).to_csv(
        audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===================================")
    print("ANNOTATION CANDIDATE SAMPLING")
    print("===================================")
    print(f"Input sentences     : {len(df):,}")
    print(f"Reports             : {len(report_ids)}")
    print(f"Per-report sample   : {args.per_report}")
    print(f"Total candidates    : {len(candidates):,}")
    print(f"Random seed         : {args.seed}")
    print(f"Candidate CSV       : {output_path}")
    print(f"Audit CSV           : {audit_path}")
    print("===================================")

    # Hard assertions for reproducibility.
    assert len(candidates) == 50 * args.per_report
    assert candidates["candidate_id"].is_unique
    assert candidates["sentence_id"].is_unique
    assert candidates.groupby("report_id").size().eq(args.per_report).all()


if __name__ == "__main__":
    main()
