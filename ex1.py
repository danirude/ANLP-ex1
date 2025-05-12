

from dataclasses import dataclass, field
from typing import Optional
import os
import numpy as np
import pandas as pd  # Excel export
import evaluate
import wandb
import torch
from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    EvalPrediction,
)

wandb.login()

# -----------------------------------------------------------------------------
# 1 · CLI arguments
# -----------------------------------------------------------------------------
@dataclass
class CustomArgs:
    max_train_samples: int = field(default=-1)
    max_eval_samples: int = field(default=-1)
    max_predict_samples: int = field(default=-1)

    lr: float = field(default=1e-4, metadata={"help": "Learning rate (default: 1e-4)"})
    num_train_epochs: int = field(default=3, metadata={"help": "Epochs (1‑5, default: 3)"})
    batch_size: int = field(default=32, metadata={"help": "Per‑device batch size (default: 32)"})

    do_train: bool = field(default=False)
    do_predict: bool = field(default=False)

    model_path: Optional[str] = field(default=None)


parser = HfArgumentParser(CustomArgs)
(custom_args,) = parser.parse_args_into_dataclasses()

if not (1 <= custom_args.num_train_epochs <= 5):
    raise ValueError("num_train_epochs must be between 1 and 5 inclusive.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------------------------------------------------------
# 2 · Dataset loading
# -----------------------------------------------------------------------------
raw_ds = load_dataset("glue", "mrpc")
if custom_args.max_train_samples != -1:
    raw_ds["train"] = raw_ds["train"].select(range(custom_args.max_train_samples))
if custom_args.max_eval_samples != -1:
    raw_ds["validation"] = raw_ds["validation"].select(range(custom_args.max_eval_samples))
if custom_args.max_predict_samples != -1:
    raw_ds["test"] = raw_ds["test"].select(range(custom_args.max_predict_samples))

# -----------------------------------------------------------------------------
# 3 · Model & tokenizer
# -----------------------------------------------------------------------------
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = None

# -----------------------------------------------------------------------------
# 4 · Tokenisation
# -----------------------------------------------------------------------------
max_len = tokenizer.model_max_length

def preprocess_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_len)

encoded = raw_ds.map(preprocess_function, batched=True, remove_columns=["sentence1", "sentence2", "idx"])
collator = DataCollatorWithPadding(tokenizer)

# -----------------------------------------------------------------------------
# 5 · Metrics
# -----------------------------------------------------------------------------
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(pred: EvalPrediction):
    preds = np.argmax(pred.predictions, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=pred.label_ids)

raw_val_s1 = raw_ds["validation"]["sentence1"]
raw_val_s2 = raw_ds["validation"]["sentence2"]
val_labels  = raw_ds["validation"]["label"]
raw_test_s1 = raw_ds["test"]["sentence1"]
raw_test_s2 = raw_ds["test"]["sentence2"]

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------
VAL_RES_FILE  = "res.txt"
TEST_RES_FILE = "test_res.txt"
model_dir = f"epoch_num_{custom_args.num_train_epochs}_lr_{custom_args.lr}_batch_size_{custom_args.batch_size}"
os.makedirs(model_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 6 · Training (optional)
# -----------------------------------------------------------------------------
if custom_args.do_train:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    train_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=custom_args.lr,
        per_device_train_batch_size=custom_args.batch_size,
        per_device_eval_batch_size=custom_args.batch_size,
        num_train_epochs=custom_args.num_train_epochs,
        logging_steps=1,
        report_to=["wandb"],
        seed=42,
    )

    with wandb.init(project="anlp_ex1", name=model_dir, config=vars(custom_args)):
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=encoded["train"],
            eval_dataset=encoded["validation"],
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        # Save **final** model & tokenizer (no intermediate checkpoints)
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)

        # ---------------- Validation metrics & Excel export ------------------
        val_output = trainer.predict(encoded["validation"], metric_key_prefix="val")
        val_preds  = np.argmax(val_output.predictions, axis=-1)
        val_acc    = (val_preds == np.array(val_labels)).mean()

        # Log to res.txt
        with open(VAL_RES_FILE, "a" if os.path.exists(VAL_RES_FILE) else "w", encoding="utf-8") as fp:
            fp.write(f"epoch_num: {custom_args.num_train_epochs}, lr: {custom_args.lr}, batch_size: {custom_args.batch_size}, eval_acc: {val_acc:.4f}\n")

        # Save Excel file with predictions + labels + correctness flag
        df_val = pd.DataFrame({
            "sentence1": raw_val_s1,
            "sentence2": raw_val_s2,
            "predicted_label": val_preds,
            "true_label": val_labels,
            "correct": (val_preds == np.array(val_labels)).astype(int),  # 1 if correct else 0
        })
        excel_path = os.path.join(model_dir, "validation_predictions.xlsx")
        df_val.to_excel(excel_path, index=False)
        print(f"→ Validation predictions saved to {excel_path}")

# -----------------------------------------------------------------------------
# 7 · Prediction (optional)
# -----------------------------------------------------------------------------
if custom_args.do_predict:
    if model is None:
        if custom_args.model_path is None:
            raise ValueError("Provide --model_path when predicting without training.")
        model = AutoModelForSequenceClassification.from_pretrained(custom_args.model_path).to(device)
    model.eval()

    pred_args = TrainingArguments(
        output_dir=model_dir,
        per_device_eval_batch_size=custom_args.batch_size,
        do_train=False,
        do_predict=True,
        report_to=["none"],
        seed=42,
    )
    predictor = Trainer(model=model, args=pred_args, tokenizer=tokenizer, data_collator=collator)

    logits = predictor.predict(encoded["test"], metric_key_prefix="test").predictions
    preds = np.argmax(logits, axis=-1)

    # Save predictions file
    pred_path = f"predictions.txt"
    with open(pred_path, "w", encoding="utf-8") as fp:
        for s1, s2, p in zip(raw_test_s1, raw_test_s2, preds):
            fp.write(f"{s1}###{s2}###{int(p)}\n")

    # Compute test accuracy if labels are available
    if "label" in encoded["test"].column_names:
        test_labels = np.array(encoded["test"]["label"])
        if not np.all(test_labels == -1):
            test_acc = (preds == test_labels).mean()
            with open(TEST_RES_FILE, "a" if os.path.exists(TEST_RES_FILE) else "w", encoding="utf-8") as fp:
                fp.write(f"epoch_num: {custom_args.num_train_epochs}, lr: {custom_args.lr}, batch_size: {custom_args.batch_size}, test_acc: {test_acc:.4f}\n")
            print(f"→ Test accuracy: {test_acc:.4f} (logged to {TEST_RES_FILE})")
        else:
            print("→ Test labels not available; skipping test accuracy logging.")

print("✓ Run complete.")
wandb.finish()
