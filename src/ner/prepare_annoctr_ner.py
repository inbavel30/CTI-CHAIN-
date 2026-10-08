from pathlib import Path
import json
import sys


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "ANNOCTR" / "ner_json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ner"


# ============================================================
# ANNOCTR NER LABELS
# ============================================================

LABELS = [
    "O",
    "B-LOC",
    "I-LOC",
    "B-ORG",
    "I-ORG",
    "B-SECTOR",
    "I-SECTOR",
]

LABEL2ID = {
    label: idx
    for idx, label in enumerate(LABELS)
}

ID2LABEL = {
    idx: label
    for label, idx in LABEL2ID.items()
}


# ============================================================
# DATA SPLITS
# ============================================================

SPLITS = {
    "train": "train.json",
    "validation": "dev.json",
    "test": "test.json",
}


# ============================================================
# VALIDATE ONE RECORD
# ============================================================

def validate_record(record, source_file, line_number):
    required_fields = [
        "id",
        "tokens",
        "text",
        "ne_tags",
    ]

    for field in required_fields:
        if field not in record:
            raise ValueError(
                f"Missing field '{field}' in "
                f"{source_file} line {line_number}"
            )

    tokens = record["tokens"]
    tags = record["ne_tags"]

    if len(tokens) != len(tags):
        raise ValueError(
            f"Token/tag length mismatch in "
            f"{source_file} line {line_number}: "
            f"{len(tokens)} tokens vs {len(tags)} tags"
        )

    for tag in tags:
        if tag not in LABEL2ID:
            raise ValueError(
                f"Unknown NER tag '{tag}' in "
                f"{source_file} line {line_number}"
            )


# ============================================================
# CONVERT ONE SPLIT
# ============================================================

def convert_split(split_name, filename):

    input_path = INPUT_DIR / filename
    output_path = OUTPUT_DIR / f"{split_name}.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    record_count = 0
    token_count = 0
    entity_token_count = 0

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as infile, open(
        output_path,
        "w",
        encoding="utf-8"
    ) as outfile:

        for line_number, line in enumerate(
            infile,
            start=1
        ):

            if not line.strip():
                continue

            record = json.loads(line)

            validate_record(
                record,
                filename,
                line_number
            )

            tokens = record["tokens"]
            tags = record["ne_tags"]

            output_record = {
                "id": record["id"],
                "tokens": tokens,
                "ner_tags": tags,
                "ner_tag_ids": [
                    LABEL2ID[tag]
                    for tag in tags
                ],
            }

            outfile.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False
                )
                + "\n"
            )

            record_count += 1
            token_count += len(tokens)

            entity_token_count += sum(
                1 for tag in tags
                if tag != "O"
            )

    return {
        "records": record_count,
        "tokens": token_count,
        "entity_tokens": entity_token_count,
        "output": output_path,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AnnoCTR NER DATA PREPARATION")
    print("=" * 70)

    print(f"Input directory : {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {}

    for split_name, filename in SPLITS.items():

        print(f"Processing {split_name}...")

        results[split_name] = convert_split(
            split_name,
            filename
        )

        result = results[split_name]

        print(
            f"  Records       : {result['records']}"
        )

        print(
            f"  Tokens        : {result['tokens']}"
        )

        print(
            f"  Entity tokens : {result['entity_tokens']}"
        )

        print(
            f"  Output        : {result['output']}"
        )

        print()

    # --------------------------------------------------------
    # Save label mapping
    # --------------------------------------------------------

    label_map = {
        "labels": LABELS,
        "label2id": LABEL2ID,
        "id2label": {
            str(k): v
            for k, v in ID2LABEL.items()
        },
        "source": "AnnoCTR",
        "source_field": "ne_tags",
    }

    label_map_path = OUTPUT_DIR / "label_map.json"

    with open(
        label_map_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            label_map,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("=" * 70)
    print("LABEL MAP")
    print("=" * 70)

    for label, idx in LABEL2ID.items():
        print(f"{idx}: {label}")

    print()
    print(f"Saved: {label_map_path}")

    print()
    print("=" * 70)
    print("PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(str(exc))
        sys.exit(1)