from pathlib import Path
import json
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "ner"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "ner"
    / "bert_base_uncased"
)

MODEL_NAME = "google-bert/bert-base-uncased"

MAX_LENGTH = 256
SEED = 42


# ============================================================
# ANNOCTR LABELS
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
# ENVIRONMENT CHECK
# ============================================================

print("=" * 80)
print("BERT NER TRAINING")
print("=" * 80)

print(f"PyTorch        : {torch.__version__}")
print(f"CUDA build     : {torch.version.cuda}")
print(f"CUDA available : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. Training stopped."
    )

GPU_NAME = torch.cuda.get_device_name(0)

GPU_MEMORY = (
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3
)

print(f"GPU            : {GPU_NAME}")
print(f"GPU VRAM       : {GPU_MEMORY:.2f} GB")
print(f"Model          : {MODEL_NAME}")
print(f"Max length     : {MAX_LENGTH}")
print(f"Seed           : {SEED}")
print()


# ============================================================
# CHECK DATA FILES
# ============================================================

print("=" * 80)
print("CHECKING DATA")
print("=" * 80)

required_files = [
    DATA_DIR / "train.jsonl",
    DATA_DIR / "validation.jsonl",
    DATA_DIR / "test.jsonl",
    DATA_DIR / "label_map.json",
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

    print(f"[OK] {file_path}")

print()


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 80)
print("LOADING ANNOCTR DATASET")
print("=" * 80)

data_files = {
    "train": str(DATA_DIR / "train.jsonl"),
    "validation": str(DATA_DIR / "validation.jsonl"),
    "test": str(DATA_DIR / "test.jsonl"),
}

dataset = load_dataset(
    "json",
    data_files=data_files,
)

print(dataset)
print()

print(
    f"Train records      : {len(dataset['train'])}"
)

print(
    f"Validation records : {len(dataset['validation'])}"
)

print(
    f"Test records       : {len(dataset['test'])}"
)

print()


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("=" * 80)
print("LOADING BERT TOKENIZER")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print(
    f"Tokenizer : {tokenizer.__class__.__name__}"
)

print()


# ============================================================
# TOKENIZATION + LABEL ALIGNMENT
# ============================================================

def tokenize_and_align_labels(examples):

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=MAX_LENGTH,
        is_split_into_words=True,
    )

    all_labels = []

    for batch_index, labels in enumerate(
        examples["ner_tag_ids"]
    ):

        word_ids = tokenized_inputs.word_ids(
            batch_index=batch_index
        )

        previous_word_id = None

        label_ids = []

        for word_id in word_ids:

            # Special tokens
            if word_id is None:

                label_ids.append(-100)

            # First subword of a word
            elif word_id != previous_word_id:

                label_ids.append(
                    labels[word_id]
                )

            # Additional subword
            else:

                label_ids.append(-100)

            previous_word_id = word_id

        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels

    return tokenized_inputs


print("=" * 80)
print("TOKENIZING DATA")
print("=" * 80)

tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing and aligning labels",
)

print(tokenized_dataset)
print()


# ============================================================
# LOAD BERT NER MODEL
# ============================================================

print("=" * 80)
print("LOADING BERT NER MODEL")
print("=" * 80)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(
    f"Model parameters : {parameter_count:,}"
)

print()


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_prediction):

    predictions, labels = eval_prediction

    predictions = np.argmax(
        predictions,
        axis=2,
    )

    true_predictions = []
    true_labels = []

    for prediction, label in zip(
        predictions,
        labels,
    ):

        current_predictions = []
        current_labels = []

        for pred_id, label_id in zip(
            prediction,
            label,
        ):

            # Ignore special/subword positions
            if label_id == -100:
                continue

            current_predictions.append(
                ID2LABEL[int(pred_id)]
            )

            current_labels.append(
                ID2LABEL[int(label_id)]
            )

        true_predictions.append(
            current_predictions
        )

        true_labels.append(
            current_labels
        )

    precision = precision_score(
        true_labels,
        true_predictions,
        average="micro",
    )

    recall = recall_score(
        true_labels,
        true_predictions,
        average="micro",
    )

    f1 = f1_score(
        true_labels,
        true_predictions,
        average="micro",
    )

    accuracy = accuracy_score(
        true_labels,
        true_predictions,
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

training_args = TrainingArguments(

    output_dir=str(OUTPUT_DIR),

    # RTX 3050 4 GB VRAM
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    # Effective batch size = 4 x 4 = 16
    gradient_accumulation_steps=4,

    learning_rate=5e-5,

    weight_decay=0.01,

    # FIRST TRAINING RUN
    num_train_epochs=1,

    eval_strategy="steps",
    eval_steps=500,

    save_strategy="steps",
    save_steps=500,

    logging_strategy="steps",
    logging_steps=100,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    # NVIDIA GPU mixed precision
    fp16=True,

    report_to="none",

    seed=SEED,

    save_total_limit=2,

    dataloader_num_workers=0,

    remove_unused_columns=False,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset["train"],

    eval_dataset=tokenized_dataset["validation"],

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,
)


# ============================================================
# TRAIN
# ============================================================

print("=" * 80)
print("STARTING BERT TRAINING")
print("=" * 80)

print(
    "GPU memory before training:"
)

print(
    f"Allocated : "
    f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
)

print(
    f"Reserved  : "
    f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB"
)

print()

train_result = trainer.train()


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 80)
print("BERT VALIDATION RESULTS")
print("=" * 80)

validation_metrics = trainer.evaluate(
    tokenized_dataset["validation"]
)

for key, value in validation_metrics.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.6f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ============================================================
# TEST
# ============================================================

print()
print("=" * 80)
print("BERT TEST RESULTS")
print("=" * 80)

test_metrics = trainer.evaluate(
    tokenized_dataset["test"]
)

for key, value in test_metrics.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.6f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("=" * 80)
print("SAVING BERT MODEL")
print("=" * 80)

trainer.save_model(
    str(OUTPUT_DIR)
)

tokenizer.save_pretrained(
    str(OUTPUT_DIR)
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": MODEL_NAME,
    "seed": SEED,
    "max_length": MAX_LENGTH,
    "train_records": len(dataset["train"]),
    "validation_records": len(dataset["validation"]),
    "test_records": len(dataset["test"]),
    "validation": validation_metrics,
    "test": test_metrics,
}

metrics_path = (
    OUTPUT_DIR / "metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=2,
    )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 80)
print("BERT NER TRAINING COMPLETED")
print("=" * 80)

print(
    f"Model directory : {OUTPUT_DIR}"
)

print(
    f"Metrics file    : {metrics_path}"
)

print()