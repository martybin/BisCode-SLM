import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "codeparrot/codeparrot-clean-train"

OUTPUT_PATH = Path("data/raw/python_code.jsonl")

MAX_SAMPLES = 100_000

ALLOWED_LICENSES = {
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "unlicense",
    "cc0-1.0",
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")

dataset = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True,
)

count = 0
skipped = 0

with OUTPUT_PATH.open("w", encoding="utf-8") as f:

    progress = tqdm(total=MAX_SAMPLES)

    for sample in dataset:

        code = sample.get("content", "")
        license_name = sample.get("license", "")

        if not isinstance(code, str):
            skipped += 1
            continue

        if license_name not in ALLOWED_LICENSES:
            skipped += 1
            continue

        if len(code.strip()) < 100:
            skipped += 1
            continue

        if len(code) > 100_000:
            skipped += 1
            continue

        record = {
            "content": code,
            "repo_name": sample.get("repo_name"),
            "path": sample.get("path"),
            "license": license_name,
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

        count += 1
        progress.update(1)

        if count >= MAX_SAMPLES:
            break

    progress.close()

print("\nDownload completed!")
print(f"Saved samples: {count:,}")
print(f"Skipped samples: {skipped:,}")
print(f"Output: {OUTPUT_PATH}")
