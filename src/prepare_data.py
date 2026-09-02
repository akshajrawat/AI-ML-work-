import os
from pathlib import Path

# Set Hugging Face cache directory to D: drive to keep C: drive clean
os.environ["HF_HOME"] = r"D:\llm-from-scratch\.cache\huggingface"

from datasets import load_dataset

print("Downloading TinyStories dataset...")

# 1. Load 50,000 stories for training
train = load_dataset(
    "roneneldan/TinyStories",
    split="train[:50000]"
)

# 2. Load 2,000 stories for validation
val = load_dataset(
    "roneneldan/TinyStories",
    split="validation[:2000]"
)

# 3. Create 'data' folder
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# 4. Join all stories together with double newlines
train_text = "\n\n".join(train["text"])
val_text = "\n\n".join(val["text"])

# 5. Save out train.txt and val.txt
(data_dir / "train.txt").write_text(
    train_text,
    encoding="utf-8"
)

(data_dir / "val.txt").write_text(
    val_text,
    encoding="utf-8"
)

print("\n" + "=" * 50)
print("DATASET PREPARATION COMPLETE:")
print("=" * 50)
print("Train stories:", len(train))
print("Validation stories:", len(val))
print("Train characters:", len(train_text))
print("Validation characters:", len(val_text))
