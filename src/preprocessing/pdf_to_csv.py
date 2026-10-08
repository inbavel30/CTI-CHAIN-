import fitz
import pandas as pd
from pathlib import Path
import re

PDF_DIR = Path("data/raw/reports/files")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # Basic sentence segmentation
    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [s.strip() for s in sentences if s.strip()]


for pdf_file in sorted(PDF_DIR.glob("*.pdf")):

    report_id = pdf_file.stem.split("_")[0]

    print(f"Processing {pdf_file.name}")

    document = fitz.open(pdf_file)

    sentence_counter = 1

    for page_number, page in enumerate(document, start=1):

        page_text = page.get_text("text")

        if not page_text.strip():
            continue

        paragraphs = re.split(r"\n\s*\n", page_text)

        for paragraph_number, paragraph in enumerate(paragraphs, start=1):

            paragraph = re.sub(r"\s+", " ", paragraph).strip()

            if not paragraph:
                continue

            sentences = split_sentences(paragraph)

            for sentence in sentences:

                rows.append({
                    "report_id": report_id,
                    "page": page_number,
                    "paragraph_id": f"{report_id}_P{paragraph_number:04d}",
                    "sentence_id": f"{report_id}_S{sentence_counter:06d}",
                    "text": sentence
                })

                sentence_counter += 1

    document.close()


df = pd.DataFrame(rows)

output_file = OUTPUT_DIR / "sentences.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print()
print("===================================")
print("PDF EXTRACTION COMPLETED")
print("===================================")
print(f"Reports processed : {df['report_id'].nunique()}")
print(f"Sentences created : {len(df)}")
print(f"Output            : {output_file}")
print("===================================")