from __future__ import annotations
import argparse, json
from pathlib import Path
from jsonschema import Draft202012Validator

BASE = Path("data/annotations")
KINDS = ("ner","events","relations","temporal","attack","evidence")

def validate(kind):
    schema = json.loads((BASE/kind/"schema.json").read_text(encoding="utf-8"))
    path = BASE/kind/"annotations.jsonl"
    v = Draft202012Validator(schema)
    count, errors = 0, []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        count += 1
        try: obj = json.loads(line)
        except Exception as e:
            errors.append(f"line {line_no}: invalid JSON: {e}"); continue
        errors += [f"line {line_no}: {e.message}" for e in v.iter_errors(obj)]
    return count, errors

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=KINDS+("all",), default="all")
    a = p.parse_args()
    kinds = KINDS if a.kind=="all" else [a.kind]
    failed=False
    for k in kinds:
        n, errors = validate(k)
        print(f"[{k}] records={n}")
        if errors:
            failed=True
            for e in errors[:20]: print("  ERROR:", e)
        else: print("  schema validation: PASSED")
    raise SystemExit(1 if failed else 0)
