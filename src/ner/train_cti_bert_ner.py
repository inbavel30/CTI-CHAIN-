from pathlib import Path
from datetime import datetime
import json
import csv
import sys
import time

import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
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
    / "cti_bert"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "ibm-research/CTI-BERT"


# ============================================================
# LABELS
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
# TRAINING CONFIGURATION
# ============================================================

MAX_LENGTH = 256

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4

GRADIENT_ACCUMULATION_STEPS = 4

EFFECTIVE_BATCH_SIZE = (
    TRAIN_BATCH_SIZE
    * GRADIENT_ACCUMULATION_STEPS
)

LEARNING_RATE = 5e-5

WEIGHT_DECAY = 0.01

NUM_EPOCHS = 3

SEED = 42


# ============================================================
# TIME
# ============================================================

EXPERIMENT_START = datetime.now()

EXPERIMENT_START_ISO = (
    EXPERIMENT_START.isoformat(
        timespec="seconds"
    )
)

GLOBAL_START = time.perf_counter()


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 90)
print("CTI-BERT NER EXPERIMENT")
print("=" * 90)

print()
print("Model:", MODEL_NAME)
print("Start time:", EXPERIMENT_START_ISO)
print()

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():

    print()
    print("ERROR: CUDA is not available.")
    print(
        "Install a CUDA-enabled PyTorch build "
        "before running this experiment."
    )

    sys.exit(1)


GPU_NAME = torch.cuda.get_device_name(0)

GPU_VRAM_GB = round(
    torch.cuda.get_device_properties(0)
    .total_memory
    / (1024 ** 3),
    2
)

print("GPU:", GPU_NAME)
print("GPU VRAM:", GPU_VRAM_GB, "GB")
print()


# ============================================================
# VERIFY DATA
# ============================================================

print("=" * 90)
print("DATASET VERIFICATION")
print("=" * 90)

required_files = [
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "label_map.json",
]

for filename in required_files:

    path = DATA_DIR / filename

    if not path.exists():

        print(
            f"ERROR: Missing file: {path}"
        )

        sys.exit(1)

    print(
        f"FOUND: {filename} "
        f"({path.stat().st_size:,} bytes)"
    )

print()


# ============================================================
# LOAD LABEL MAP
# ============================================================

with open(
    DATA_DIR / "label_map.json",
    "r",
    encoding="utf-8"
) as f:

    label_map = json.load(f)


print("NER LABELS:")

for idx, label in enumerate(LABELS):

    print(
        f"{idx}: {label}"
    )

print()


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 90)
print("LOADING DATASET")
print("=" * 90)

data_files = {

    "train":
        str(DATA_DIR / "train.jsonl"),

    "validation":
        str(DATA_DIR / "validation.jsonl"),

    "test":
        str(DATA_DIR / "test.jsonl"),
}

dataset = load_dataset(
    "json",
    data_files=data_files,
)

print(dataset)

TRAIN_RECORDS = len(
    dataset["train"]
)

VALIDATION_RECORDS = len(
    dataset["validation"]
)

TEST_RECORDS = len(
    dataset["test"]
)

print()
print("Train records:",
      TRAIN_RECORDS)

print("Validation records:",
      VALIDATION_RECORDS)

print("Test records:",
      TEST_RECORDS)

print()


# ============================================================
# ORIGINAL TOKEN COUNTS
# ============================================================

TRAIN_TOKENS = sum(
    len(x["tokens"])
    for x in dataset["train"]
)

VALIDATION_TOKENS = sum(
    len(x["tokens"])
    for x in dataset["validation"]
)

TEST_TOKENS = sum(
    len(x["tokens"])
    for x in dataset["test"]
)

print("Train tokens:",
      TRAIN_TOKENS)

print("Validation tokens:",
      VALIDATION_TOKENS)

print("Test tokens:",
      TEST_TOKENS)

print()


# ============================================================
# DATA PREPROCESSING / TOKENIZATION
# ============================================================

print("=" * 90)
print("DATA PREPROCESSING / TOKENIZATION")
print("=" * 90)

PREPROCESSING_START = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


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

        word_ids = (
            tokenized_inputs
            .word_ids(
                batch_index=batch_index
            )
        )

        previous_word_id = None

        label_ids = []

        for word_id in word_ids:

            if word_id is None:

                label_ids.append(-100)

            elif word_id != previous_word_id:

                label_ids.append(
                    labels[word_id]
                )

            else:

                label_ids.append(-100)

            previous_word_id = word_id

        all_labels.append(
            label_ids
        )

    tokenized_inputs["labels"] = (
        all_labels
    )

    return tokenized_inputs


tokenized_dataset = dataset.map(

    tokenize_and_align_labels,

    batched=True,

    remove_columns=(
        dataset["train"].column_names
    ),

    desc="Tokenizing CTI-BERT dataset",
)


PREPROCESSING_END = time.perf_counter()

PREPROCESSING_TIME = (
    PREPROCESSING_END
    - PREPROCESSING_START
)

print()
print(
    "Data preprocessing/tokenization time:",
    f"{PREPROCESSING_TIME:.2f} seconds"
)

print()


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 90)
print("LOADING CTI-BERT")
print("=" * 90)

model = AutoModelForTokenClassification.from_pretrained(

    MODEL_NAME,

    num_labels=len(LABELS),

    id2label=ID2LABEL,

    label2id=LABEL2ID,
)

PARAMETERS = sum(
    p.numel()
    for p in model.parameters()
)

TRAINABLE_PARAMETERS = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "Total parameters:",
    f"{PARAMETERS:,}"
)

print(
    "Trainable parameters:",
    f"{TRAINABLE_PARAMETERS:,}"
)

print()


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = (
    DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    eval_prediction
):

    predictions, labels = (
        eval_prediction
    )

    predictions = np.argmax(
        predictions,
        axis=2
    )

    true_predictions = []

    true_labels = []

    for prediction, label in zip(
        predictions,
        labels
    ):

        current_predictions = []

        current_labels = []

        for pred_id, label_id in zip(
            prediction,
            label
        ):

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
        average="micro"
    )

    recall = recall_score(
        true_labels,
        true_predictions,
        average="micro"
    )

    f1 = f1_score(
        true_labels,
        true_predictions,
        average="micro"
    )

    accuracy = accuracy_score(
        true_labels,
        true_predictions
    )

    return {

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "accuracy": accuracy,
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=str(
        OUTPUT_DIR
    ),

    per_device_train_batch_size=(
        TRAIN_BATCH_SIZE
    ),

    per_device_eval_batch_size=(
        EVAL_BATCH_SIZE
    ),

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    num_train_epochs=NUM_EPOCHS,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="epoch",

    # IMPORTANT:
    # Best model is selected using
    # validation LOSS, not accuracy.

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    fp16=True,

    report_to="none",

    seed=SEED,

    save_total_limit=3,

    dataloader_num_workers=0,

    remove_unused_columns=False,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=(
        tokenized_dataset["train"]
    ),

    eval_dataset=(
        tokenized_dataset["validation"]
    ),

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,
)


# ============================================================
# TRAINING
# ============================================================

print("=" * 90)
print("STARTING TRAINING")
print("=" * 90)

print()
print("Epochs:",
      NUM_EPOCHS)

print("Train batch:",
      TRAIN_BATCH_SIZE)

print("Gradient accumulation:",
      GRADIENT_ACCUMULATION_STEPS)

print("Effective batch:",
      EFFECTIVE_BATCH_SIZE)

print("Learning rate:",
      LEARNING_RATE)

print()

TRAINING_START = time.perf_counter()

train_output = trainer.train()

TRAINING_END = time.perf_counter()

TRAINING_TIME = (
    TRAINING_END
    - TRAINING_START
)

print()
print(
    "Training runtime:",
    f"{TRAINING_TIME:.2f} seconds"
)

print()


# ============================================================
# TRAINING HISTORY
# ============================================================

log_history = trainer.state.log_history


# ============================================================
# EXTRACT EPOCH RESULTS
# ============================================================

epoch_results = []

for log in log_history:

    if "epoch" not in log:
        continue

    if (
        "loss" in log
        and "eval_loss" not in log
    ):

        epoch_results.append({

            "epoch":
                float(log["epoch"]),

            "training_loss":
                float(log["loss"]),

        })


    elif "eval_loss" in log:

        # Find corresponding training loss
        # for this epoch.

        epoch_number = (
            float(log["epoch"])
        )

        matching_train = [

            x

            for x in epoch_results

            if abs(
                x["epoch"]
                - epoch_number
            ) < 1e-6
        ]

        if matching_train:

            matching_train[
                -1
            ][
                "validation_loss"
            ] = float(
                log["eval_loss"]
            )

        else:

            epoch_results.append({

                "epoch":
                    epoch_number,

                "validation_loss":
                    float(
                        log["eval_loss"]
                    ),
            })


# ============================================================
# SELECT BEST EPOCH BY VALIDATION LOSS
# ============================================================

valid_epoch_results = [

    x

    for x in epoch_results

    if "validation_loss" in x
]

if not valid_epoch_results:

    print(
        "WARNING: Could not extract "
        "epoch validation losses."
    )

    BEST_EPOCH = None
    BEST_VALIDATION_LOSS = None

else:

    best_epoch_record = min(

        valid_epoch_results,

        key=lambda x:
            x["validation_loss"]
    )

    BEST_EPOCH = (
        best_epoch_record["epoch"]
    )

    BEST_VALIDATION_LOSS = (
        best_epoch_record[
            "validation_loss"
        ]
    )


# ============================================================
# FIND BEST EPOCH TRAINING LOSS
# ============================================================

BEST_TRAINING_LOSS = None

if BEST_EPOCH is not None:

    for record in valid_epoch_results:

        if (
            record["epoch"]
            == BEST_EPOCH
        ):

            BEST_TRAINING_LOSS = (
                record.get(
                    "training_loss"
                )
            )

            break


# ============================================================
# VALIDATION PERFORMANCE
# ============================================================

print("=" * 90)
print("BEST EPOCH / VALIDATION")
print("=" * 90)

validation_metrics = trainer.evaluate(
    tokenized_dataset["validation"]
)

print()

for key, value in (
    validation_metrics.items()
):

    if isinstance(value, float):

        print(
            f"{key}: {value:.6f}"
        )

    else:

        print(
            f"{key}: {value}"
        )

print()

print(
    "Best epoch selected by validation loss:",
    BEST_EPOCH
)

print(
    "Best validation loss:",
    BEST_VALIDATION_LOSS
)

print()


# ============================================================
# TEST PERFORMANCE
# ============================================================

print("=" * 90)
print("TEST PERFORMANCE")
print("=" * 90)

TEST_START = time.perf_counter()

test_metrics = trainer.evaluate(
    tokenized_dataset["test"]
)

TEST_END = time.perf_counter()

TEST_RUNTIME = (
    TEST_END
    - TEST_START
)

print()

for key, value in (
    test_metrics.items()
):

    if isinstance(value, float):

        print(
            f"{key}: {value:.6f}"
        )

    else:

        print(
            f"{key}: {value}"
        )

print()


# ============================================================
# END TIME
# ============================================================

EXPERIMENT_END = datetime.now()

EXPERIMENT_END_ISO = (
    EXPERIMENT_END.isoformat(
        timespec="seconds"
    )
)

TOTAL_RUNTIME = (
    time.perf_counter()
    - GLOBAL_START
)


# ============================================================
# SAMPLES / SECOND
# ============================================================

# Training samples processed divided
# by actual training runtime.

SAMPLES_PER_SECOND = (
    TRAIN_RECORDS * NUM_EPOCHS
) / TRAINING_TIME


# ============================================================
# SAVE MODEL
# ============================================================

print("=" * 90)
print("SAVING MODEL")
print("=" * 90)

trainer.save_model(
    str(OUTPUT_DIR)
)

tokenizer.save_pretrained(
    str(OUTPUT_DIR)
)

print(
    "Model saved:",
    OUTPUT_DIR
)

print()


# ============================================================
# COMPLETE EXPERIMENT RECORD
# ============================================================

results = {

    # --------------------------------------------------------
    # EXPERIMENT
    # --------------------------------------------------------

    "model_name":
        MODEL_NAME,

    "experiment_start_time":
        EXPERIMENT_START_ISO,

    "experiment_end_time":
        EXPERIMENT_END_ISO,

    "total_runtime_seconds":
        round(
            TOTAL_RUNTIME,
            4
        ),

    "total_training_time_seconds":
        round(
            TRAINING_TIME,
            4
        ),

    "data_preprocessing_time_seconds":
        round(
            PREPROCESSING_TIME,
            4
        ),

    "test_evaluation_time_seconds":
        round(
            TEST_RUNTIME,
            4
        ),

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    "dataset":
        "AnnoCTR",

    "task":
        "NER",

    "train_records":
        TRAIN_RECORDS,

    "validation_records":
        VALIDATION_RECORDS,

    "test_records":
        TEST_RECORDS,

    "train_tokens":
        TRAIN_TOKENS,

    "validation_tokens":
        VALIDATION_TOKENS,

    "test_tokens":
        TEST_TOKENS,

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    "number_of_parameters":
        PARAMETERS,

    "trainable_parameters":
        TRAINABLE_PARAMETERS,

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    "number_of_epochs":
        NUM_EPOCHS,

    "best_epoch":
        BEST_EPOCH,

    "best_epoch_selection":
        "Minimum validation loss",

    "best_training_loss":
        BEST_TRAINING_LOSS,

    "best_validation_loss":
        BEST_VALIDATION_LOSS,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "max_sequence_length":
        MAX_LENGTH,

    "train_batch_size":
        TRAIN_BATCH_SIZE,

    "gradient_accumulation_steps":
        GRADIENT_ACCUMULATION_STEPS,

    "effective_batch_size":
        EFFECTIVE_BATCH_SIZE,

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    "validation_loss":
        validation_metrics.get(
            "eval_loss"
        ),

    "validation_accuracy":
        validation_metrics.get(
            "eval_accuracy"
        ),

    "validation_precision":
        validation_metrics.get(
            "eval_precision"
        ),

    "validation_recall":
        validation_metrics.get(
            "eval_recall"
        ),

    "validation_f1":
        validation_metrics.get(
            "eval_f1"
        ),

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    "test_loss":
        test_metrics.get(
            "eval_loss"
        ),

    "test_accuracy":
        test_metrics.get(
            "eval_accuracy"
        ),

    "test_precision":
        test_metrics.get(
            "eval_precision"
        ),

    "test_recall":
        test_metrics.get(
            "eval_recall"
        ),

    "test_f1":
        test_metrics.get(
            "eval_f1"
        ),

    # --------------------------------------------------------
    # HARDWARE
    # --------------------------------------------------------

    "gpu":
        GPU_NAME,

    "gpu_vram_gb":
        GPU_VRAM_GB,

    "pytorch_version":
        torch.__version__,

    "cuda_version":
        torch.version.cuda,

    # --------------------------------------------------------
    # SPEED
    # --------------------------------------------------------

    "runtime_seconds":
        round(
            TRAINING_TIME,
            4
        ),

    "samples_per_second":
        round(
            SAMPLES_PER_SECOND,
            4
        ),
}


# ============================================================
# SAVE JSON
# ============================================================

JSON_PATH = (
    OUTPUT_DIR / "metrics.json"
)

with open(
    JSON_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


# ============================================================
# SAVE CSV
# ============================================================

CSV_PATH = (
    OUTPUT_DIR / "results.csv"
)

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            results.keys()
        )
    )

    writer.writeheader()

    writer.writerow(
        results
    )


# ============================================================
# SAVE EPOCH HISTORY CSV
# ============================================================

EPOCH_CSV_PATH = (
    OUTPUT_DIR
    / "epoch_history.csv"
)

if epoch_results:

    epoch_fields = set()

    for record in epoch_results:

        epoch_fields.update(
            record.keys()
        )

    epoch_fields = list(
        epoch_fields
    )

    with open(
        EPOCH_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=epoch_fields
        )

        writer.writeheader()

        for record in epoch_results:

            writer.writerow(
                record
            )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 90)
print("CTI-BERT EXPERIMENT COMPLETED")
print("=" * 90)

print()

print("Model:")
print(MODEL_NAME)

print()

print("Start time:")
print(EXPERIMENT_START_ISO)

print()

print("End time:")
print(EXPERIMENT_END_ISO)

print()

print(
    "Total training time:",
    f"{TRAINING_TIME:.2f} seconds"
)

print(
    "Data preprocessing time:",
    f"{PREPROCESSING_TIME:.2f} seconds"
)

print()

print("Train records:",
      TRAIN_RECORDS)

print("Validation records:",
      VALIDATION_RECORDS)

print("Test records:",
      TEST_RECORDS)

print()

print(
    "Parameters:",
    f"{PARAMETERS:,}"
)

print()

print(
    "Best epoch:",
    BEST_EPOCH
)

print(
    "Best validation loss:",
    BEST_VALIDATION_LOSS
)

print()

print("TEST PERFORMANCE")

print(
    "Loss:",
    f"{test_metrics['eval_loss']:.6f}"
)

print(
    "Accuracy:",
    f"{test_metrics['eval_accuracy']:.6f}"
)

print(
    "Precision:",
    f"{test_metrics['eval_precision']:.6f}"
)

print(
    "Recall:",
    f"{test_metrics['eval_recall']:.6f}"
)

print(
    "F1:",
    f"{test_metrics['eval_f1']:.6f}"
)

print()

print(
    "Samples/second:",
    f"{SAMPLES_PER_SECOND:.4f}"
)

print()

print("Saved files:")

print(
    "Model:",
    OUTPUT_DIR
)

print(
    "JSON:",
    JSON_PATH
)

print(
    "CSV:",
    CSV_PATH
)

print(
    "Epoch history:",
    EPOCH_CSV_PATH
)

print("=" * 90)