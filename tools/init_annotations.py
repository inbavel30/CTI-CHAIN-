from pathlib import Path
ROOT = Path("data/annotations")
for kind in ("ner","events","relations","temporal","attack","evidence"):
    p = ROOT/kind
    p.mkdir(parents=True, exist_ok=True)
    (p/"annotations.jsonl").touch(exist_ok=True)
print("Annotation folders initialized.")
