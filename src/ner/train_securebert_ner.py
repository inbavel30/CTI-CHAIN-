# ============================================================
# SECUREBERT NER TRAINING (corrected)
# Evidence-Grounded CTI NER Benchmark
#
# Fixes compared to the previous version:
#   1. ner_tags can be label STRINGS ("O", "B-LOC") or integer IDs.
#      Strings are converted with label2id.
#   2. label_map.json is parsed in every common format.
#   3. Tokenizer uses add_prefix_space=True (needed for RoBERTa
#      tokenizers with is_split_into_words=True).
#   4. seqeval metrics receive label STRINGS (not integer IDs).
#   5. Trainer uses processing_class (new) or tokenizer (old)
#      depending on the installed transformers version.
#
# Requirements:
#   pip install transformers==4.57.6 datasets seqeval pandas accelerate
# ============================================================

import os
import json
import time
import random
import inspect
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = r"C:\Users\inbav\OneDrive\Desktop\CTI_ATTACK_CHAIN_PROJECT"

NER_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "ner")

TRAIN_FILE = os.path.join(NER_DATA_DIR, "train.jsonl")
VALIDATION_FILE = os.path.join(NER_DATA_DIR, "validation.jsonl")
TEST_FILE = os.path.join(NER_DATA_DIR, "test.jsonl")
LABEL_MAP_FILE = os.path.join(NER_DATA_DIR, "label_map.json")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "ner", "securebert")

MODEL_NAME = "ehsanaghaei/SecureBERT"

MAX_LENGTH = 256

# RTX 3050 (4 GB)
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3
RANDOM_SEED = 42

USE_FP16 = torch.cuda.is_available()
NUM_WORKERS = 0


# ============================================================
# UTILITIES
# ============================================================

def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def _is_int_like(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, str):
        try:
            int(value)
            return True
        except ValueError:
            return False
    return False


# ============================================================
# LABEL MAP LOADER (handles all common formats)
# ============================================================

def _mapping_to_id2label(mapping):
    """
    Accepts either {"0": "O", "1": "B-LOC"} (id -> label)
    or {"O": 0, "B-LOC": 1} (label -> id) and returns id2label.
    """
    if not isinstance(mapping, dict) or len(mapping) == 0:
        return None

    keys = list(mapping.keys())
    values = list(mapping.values())

    # id -> label
    if all(_is_int_like(k) for k in keys) and all(isinstance(v, str) for v in values):
        return {int(k): v for k, v in mapping.items()}

    # label -> id
    if all(isinstance(k, str) for k in keys) and all(_is_int_like(v) for v in values):
        return {int(v): k for k, v in mapping.items()}

    return None


def load_label_map(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = None

    # Plain list: ["O", "B-LOC", ...]
    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        result = {i: label for i, label in enumerate(data)}

    elif isinstance(data, dict):
        # Wrapped formats
        for key in ("id2label", "label2id", "label_to_id", "id_to_label",
                    "label_map", "labels"):
            if key in data:
                inner = data[key]
                if isinstance(inner, list) and all(isinstance(x, str) for x in inner):
                    result = {i: label for i, label in enumerate(inner)}
                else:
                    result = _mapping_to_id2label(inner)
                if result:
                    break

        # Plain dict formats
        if not result:
            result = _mapping_to_id2label(data)

    if not result:
        raise ValueError(f"Unsupported label_map.json format:\n{data}")

    result = dict(sorted(result.items()))

    expected_ids = list(range(len(result)))
    if list(result.keys()) != expected_ids:
        raise ValueError(
            f"Label IDs must be contiguous starting at 0. Got: {list(result.keys())}"
        )

    return result


# ============================================================
# JSONL LOADER
# ============================================================

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ============================================================
# DATA NORMALIZATION
# (accepts string labels OR integer IDs)
# ============================================================

def normalize_records(records, label2id, split_name=""):
    num_labels = len(label2id)
    normalized = []

    for row_index, record in enumerate(records):

        tokens = record.get("tokens")
        if tokens is None:
            tokens = record.get("words")
        if tokens is None:
            raise ValueError(
                f"[{split_name}] record {row_index}: no 'tokens' or 'words' field."
            )

        ner_tags = record.get("ner_tags")
        if ner_tags is None:
            ner_tags = record.get("labels")
        if ner_tags is None:
            raise ValueError(
                f"[{split_name}] record {row_index}: no 'ner_tags' or 'labels' field."
            )

        converted = []

        for tag in ner_tags:

            if isinstance(tag, bool):
                raise ValueError(
                    f"[{split_name}] record {row_index}: invalid tag {tag!r}"
                )

            if isinstance(tag, int):
                tag_id = tag

            elif isinstance(tag, str):
                if tag in label2id:
                    tag_id = label2id[tag]
                elif tag.strip() in label2id:
                    tag_id = label2id[tag.strip()]
                elif _is_int_like(tag):
                    tag_id = int(tag)
                else:
                    raise ValueError(
                        f"[{split_name}] record {row_index}: unknown label {tag!r}. "
                        f"Known labels: {list(label2id)}"
                    )
            else:
                raise ValueError(
                    f"[{split_name}] record {row_index}: unsupported tag type "
                    f"{type(tag).__name__}: {tag!r}"
                )

            if not 0 <= tag_id < num_labels:
                raise ValueError(
                    f"[{split_name}] record {row_index}: label id {tag_id} "
                    f"out of range 0..{num_labels - 1}"
                )

            converted.append(tag_id)

        if len(tokens) != len(converted):
            raise ValueError(
                f"[{split_name}] record {row_index}: "
                f"{len(tokens)} tokens vs {len(converted)} labels"
            )

        # Skip empty examples (they break tokenization alignment)
        if len(tokens) == 0:
            continue

        normalized.append(
            {
                "tokens": [str(t) for t in tokens],
                "ner_tags": converted,
            }
        )

    return normalized


# ============================================================
# TOKENIZATION + LABEL ALIGNMENT
# ============================================================

def tokenize_dataset(dataset, tokenizer):
    original_columns = dataset.column_names

    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

        all_labels = []

        for i in range(len(batch["tokens"])):
            word_ids = tokenized.word_ids(batch_index=i)
            original_labels = batch["ner_tags"][i]

            aligned = []
            previous_word_id = None

            for word_id in word_ids:
                if word_id is None:
                    aligned.append(-100)                      # special token
                elif word_id != previous_word_id:
                    aligned.append(original_labels[word_id])  # first sub-token
                else:
                    aligned.append(-100)                      # other sub-tokens
                previous_word_id = word_id

            all_labels.append(aligned)

        tokenized["labels"] = all_labels
        return tokenized

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=original_columns,
        desc="Tokenizing",
    )


# ============================================================
# METRICS (seqeval needs label STRINGS)
# ============================================================

def make_compute_metrics(id2label):

    def compute_metrics(eval_prediction):
        predictions, labels = eval_prediction

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.argmax(predictions, axis=2)

        true_predictions = []
        true_labels = []

        for prediction_row, label_row in zip(predictions, labels):
            pred_seq = []
            label_seq = []

            for prediction, label in zip(prediction_row, label_row):
                if label == -100:
                    continue
                pred_seq.append(id2label[int(prediction)])
                label_seq.append(id2label[int(label)])

            true_predictions.append(pred_seq)
            true_labels.append(label_seq)

        return {
            "precision": float(precision_score(true_labels, true_predictions, zero_division=0)),
            "recall": float(recall_score(true_labels, true_predictions, zero_division=0)),
            "f1": float(f1_score(true_labels, true_predictions, zero_division=0)),
            "accuracy": float(accuracy_score(true_labels, true_predictions)),
        }

    return compute_metrics


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.time()
    start_datetime = datetime.now().astimezone()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(RANDOM_SEED)

    # ---------------- HEADER ----------------
    print_section("SECUREBERT NER TRAINING")
    print(f"Start time: {start_datetime.isoformat()}")
    print(f"Model: {MODEL_NAME}")

    # ---------------- HARDWARE ----------------
    print_section("HARDWARE")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU memory: {gpu_memory:.2f} GB")
    else:
        print("GPU: CPU only")

    print(f"Platform: {platform.platform()}")

    # ---------------- FILE CHECK ----------------
    print_section("CHECKING INPUT FILES")

    for path in [TRAIN_FILE, VALIDATION_FILE, TEST_FILE, LABEL_MAP_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found:\n{path}")
        print(f"FOUND: {os.path.basename(path)}")

    # ---------------- LABEL MAP ----------------
    print_section("LABEL MAP")

    id2label = load_label_map(LABEL_MAP_FILE)
    label2id = {label: index for index, label in id2label.items()}
    num_labels = len(id2label)

    for index, label in id2label.items():
        print(f"{index}: {label}")

    print(f"Number of labels: {num_labels}")

    # ---------------- LOAD DATA ----------------
    print_section("LOADING DATA")

    t0 = time.time()
    train_raw = load_jsonl(TRAIN_FILE)
    validation_raw = load_jsonl(VALIDATION_FILE)
    test_raw = load_jsonl(TEST_FILE)

    print(f"Train records: {len(train_raw):,}")
    print(f"Validation records: {len(validation_raw):,}")
    print(f"Test records: {len(test_raw):,}")

    train_records = normalize_records(train_raw, label2id, "train")
    validation_records = normalize_records(validation_raw, label2id, "validation")
    test_records = normalize_records(test_raw, label2id, "test")

    data_loading_time = time.time() - t0

    train_token_count = sum(len(r["tokens"]) for r in train_records)
    validation_token_count = sum(len(r["tokens"]) for r in validation_records)
    test_token_count = sum(len(r["tokens"]) for r in test_records)

    print()
    print(f"Train tokens: {train_token_count:,}")
    print(f"Validation tokens: {validation_token_count:,}")
    print(f"Test tokens: {test_token_count:,}")

    # ---------------- LABEL DISTRIBUTION ----------------
    print_section("TRAINING LABEL DISTRIBUTION")

    label_counts = {index: 0 for index in id2label}
    for record in train_records:
        for label_id in record["ner_tags"]:
            label_counts[label_id] += 1

    for label_id in sorted(label_counts):
        print(f"{id2label[label_id]:<10}: {label_counts[label_id]:,}")

    # ---------------- DATASETS ----------------
    print_section("CREATING DATASETS")

    train_dataset = Dataset.from_list(train_records)
    validation_dataset = Dataset.from_list(validation_records)
    test_dataset = Dataset.from_list(test_records)

    print(f"Train: {len(train_dataset):,}")
    print(f"Validation: {len(validation_dataset):,}")
    print(f"Test: {len(test_dataset):,}")

    # ---------------- TOKENIZER ----------------
    print_section("LOADING SECUREBERT TOKENIZER")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        add_prefix_space=True,
    )
    tokenizer_loading_time = time.time() - t0

    print(f"Tokenizer: {tokenizer.__class__.__name__}")
    print(f"add_prefix_space: {getattr(tokenizer, 'add_prefix_space', None)}")
    print(f"Tokenizer loading time: {tokenizer_loading_time:.3f} seconds")

    # ---------------- PREPROCESSING ----------------
    print_section("PREPROCESSING")

    t0 = time.time()
    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    tokenized_validation = tokenize_dataset(validation_dataset, tokenizer)
    tokenized_test = tokenize_dataset(test_dataset, tokenizer)
    preprocessing_time = time.time() - t0

    print(f"Preprocessing time: {preprocessing_time:.3f} seconds")
    print(f"Tokenized columns: {tokenized_train.column_names}")

    for column in ["input_ids", "attention_mask", "labels"]:
        if column not in tokenized_train.column_names:
            raise RuntimeError(f"Missing tokenized column: {column}")

    # ---------------- MODEL ----------------
    print_section("LOADING SECUREBERT MODEL")

    t0 = time.time()
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model_loading_time = time.time() - t0
    parameter_count = count_parameters(model)

    print(f"Parameters: {parameter_count:,}")
    print(f"Model loading time: {model_loading_time:.3f} seconds")

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
    )

    # ---------------- TRAINING CONFIG ----------------
    print_section("TRAINING CONFIGURATION")

    effective_batch_size = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Train batch size: {TRAIN_BATCH_SIZE}")
    print(f"Eval batch size: {EVAL_BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Max sequence length: {MAX_LENGTH}")
    print(f"FP16: {USE_FP16}")
    print(f"Seed: {RANDOM_SEED}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=USE_FP16,
        dataloader_num_workers=NUM_WORKERS,
        save_total_limit=2,
        report_to="none",
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        remove_unused_columns=True,
        logging_first_step=True,
    )

    # transformers >= 4.46 uses processing_class; older uses tokenizer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(id2label),
    )

    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    # ---------------- TRAIN ----------------
    print_section("STARTING TRAINING")

    training_start = time.time()
    train_result = trainer.train()
    training_runtime = time.time() - training_start

    # ---------------- BEST CHECKPOINT ----------------
    print_section("BEST MODEL SELECTION")

    best_checkpoint = trainer.state.best_model_checkpoint
    print(f"Best checkpoint: {best_checkpoint}")

    log_history = trainer.state.log_history

    history_rows = []
    for item in log_history:
        if "epoch" not in item:
            continue
        history_rows.append(
            {
                "epoch": item.get("epoch"),
                "step": item.get("step"),
                "loss": item.get("loss"),
                "eval_loss": item.get("eval_loss"),
                "eval_precision": item.get("eval_precision"),
                "eval_recall": item.get("eval_recall"),
                "eval_f1": item.get("eval_f1"),
                "eval_accuracy": item.get("eval_accuracy"),
                "learning_rate": item.get("learning_rate"),
            }
        )

    history_df = pd.DataFrame(history_rows)

    log_history_path = os.path.join(OUTPUT_DIR, "trainer_log_history.csv")
    pd.DataFrame(log_history).to_csv(log_history_path, index=False)

    evaluation_history = [i for i in log_history if "eval_loss" in i]
    if not evaluation_history:
        raise RuntimeError("No evaluation history containing eval_loss was found.")

    best_eval_record = min(evaluation_history, key=lambda i: i["eval_loss"])
    best_epoch = best_eval_record.get("epoch")
    best_validation_loss = best_eval_record.get("eval_loss")

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_validation_loss:.6f}")

    # ---------------- TEST EVALUATION ----------------
    print_section("FINAL TEST EVALUATION")

    test_start = time.time()
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_test,
        metric_key_prefix="test",
    )
    test_runtime = time.time() - test_start

    test_loss = float(test_metrics.get("test_loss", 0.0))
    test_precision = float(test_metrics.get("test_precision", 0.0))
    test_recall = float(test_metrics.get("test_recall", 0.0))
    test_f1 = float(test_metrics.get("test_f1", 0.0))
    test_accuracy = float(test_metrics.get("test_accuracy", 0.0))

    train_metrics = train_result.metrics
    training_loss = float(train_metrics.get("train_loss", 0.0))
    trainer_runtime = float(train_metrics.get("train_runtime", training_runtime))
    samples_per_second = float(train_metrics.get("train_samples_per_second", 0.0))
    steps_per_second = float(train_metrics.get("train_steps_per_second", 0.0))

    end_datetime = datetime.now().astimezone()
    total_runtime = time.time() - overall_start

    # ---------------- SAVE MODEL ----------------
    print_section("SAVING BEST MODEL")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to:\n{OUTPUT_DIR}")

    epoch_history_path = os.path.join(OUTPUT_DIR, "epoch_history.csv")
    if not history_df.empty:
        history_df.to_csv(epoch_history_path, index=False)

    # ---------------- RESULTS ----------------
    results = {
        "model_name": MODEL_NAME,
        "experiment": "SecureBERT NER",
        "task": "Named Entity Recognition on AnnoCTR LOC / ORG / SECTOR labels",
        "start_time": start_datetime.isoformat(),
        "end_time": end_datetime.isoformat(),

        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",

        "train_samples": len(train_records),
        "validation_samples": len(validation_records),
        "test_samples": len(test_records),
        "train_tokens": train_token_count,
        "validation_tokens": validation_token_count,
        "test_tokens": test_token_count,
        "num_labels": num_labels,
        "labels": [id2label[i] for i in sorted(id2label)],

        "num_parameters": parameter_count,
        "model_loading_time_seconds": model_loading_time,
        "tokenizer_loading_time_seconds": tokenizer_loading_time,
        "data_loading_time_seconds": data_loading_time,
        "preprocessing_time_seconds": preprocessing_time,
        "max_sequence_length": MAX_LENGTH,

        "epochs": NUM_EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": effective_batch_size,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "fp16": USE_FP16,
        "random_seed": RANDOM_SEED,

        "training_loss": training_loss,
        "training_runtime_seconds": training_runtime,
        "trainer_runtime_seconds": trainer_runtime,
        "samples_per_second": samples_per_second,
        "steps_per_second": steps_per_second,

        "best_epoch": float(best_epoch) if best_epoch is not None else None,
        "best_validation_loss": float(best_validation_loss),
        "best_checkpoint": best_checkpoint,

        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_runtime_seconds": test_runtime,

        "total_runtime_seconds": total_runtime,
        "output_directory": OUTPUT_DIR,
    }

    metrics_json_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    results_csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    pd.DataFrame([results]).to_csv(results_csv_path, index=False)

    test_metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # ---------------- FINAL REPORT ----------------
    print_section("SECUREBERT FINAL RESULTS")

    print(f"Model: {MODEL_NAME}")
    print(f"Parameters: {parameter_count:,}")
    print()
    print(f"Start time: {start_datetime.isoformat()}")
    print(f"End time: {end_datetime.isoformat()}")
    print()
    print(f"Preprocessing time: {preprocessing_time:.3f} sec")
    print(f"Training runtime: {training_runtime:.3f} sec")
    print(f"Total runtime: {total_runtime:.3f} sec")
    print()
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print()
    print(f"Training loss: {training_loss:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test accuracy: {test_accuracy:.6f}")
    print(f"Test precision: {test_precision:.6f}")
    print(f"Test recall: {test_recall:.6f}")
    print(f"Test F1: {test_f1:.6f}")
    print()
    print(f"Samples/sec: {samples_per_second:.4f}")
    print()
    print("OUTPUT FILES:")
    print(f"Model:             {OUTPUT_DIR}")
    print(f"Metrics JSON:      {metrics_json_path}")
    print(f"Results CSV:       {results_csv_path}")
    print(f"Epoch history CSV: {epoch_history_path}")
    print(f"Trainer log CSV:   {log_history_path}")
    print(f"Test metrics JSON: {test_metrics_path}")
    print()
    print("=" * 80)
    print("SECUREBERT TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()