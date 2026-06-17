"""
BERT fine-tuning for Circa indirect answer classification.
Supports two modes:
  - baseline: BERT-MNLI fine-tuned on [question + answer]
  - augmented: BERT-MNLI fine-tuned on [question + answer + LLM explanation]
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from dataset import (CircaDataset, CircaAugmentedDataset, CircaRepeatDataset,
                     CircaTemplateDataset, CircaOracleDataset, create_splits)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

DEFAULT_MODEL = "textattack/bert-base-uncased-MNLI"
NUM_CLASSES = 6
LABEL_NAMES = [
    "Yes", "Probably yes", "Yes, conditional",
    "No", "Probably no", "In the middle"
]

# Maps mode -> (DatasetClass, max_length, data_filename)
MODE_CONFIG = {
    "baseline": (CircaDataset,          128, "circa_processed.csv"),
    "repeat":   (CircaRepeatDataset,    192, "circa_processed.csv"),
    "template": (CircaTemplateDataset,  192, "circa_processed.csv"),
    "oracle":   (CircaOracleDataset,    192, "circa_processed.csv"),
    "llm":      (CircaAugmentedDataset, 256, "circa_with_explanations.csv"),
}


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )

        total_loss += outputs.loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=LABEL_NAMES,
                                   labels=list(range(NUM_CLASSES)),
                                   output_dict=True, zero_division=0)
    return avg_loss, accuracy, report, all_preds, all_labels


def run_training(args):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    DatasetClass, max_len, data_fname = MODE_CONFIG[args.mode]
    data_file = os.path.join(DATA_DIR, data_fname)
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        if args.mode == "llm":
            print("  Run src/generate_explanations.py --provider azure first.")
        return

    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} examples from {data_file}")

    # For fair comparison: non-LLM modes are capped to the same size as the LLM dataset
    # unless --max_train is explicitly set to something else or 0 (0 = no cap)
    if args.mode != "llm" and getattr(args, "max_train", None) is None:
        llm_file = os.path.join(DATA_DIR, "circa_with_explanations.csv")
        if os.path.exists(llm_file):
            llm_size = len(pd.read_csv(llm_file))
            if len(df) > llm_size:
                df = df.sample(n=llm_size, random_state=args.seed)
                print(f"Auto-capped to {llm_size} examples to match LLM dataset size (fair comparison)")

    # Create splits
    train_df, val_df, test_df = create_splits(df, test_scenario=args.test_scenario)
    if getattr(args, "max_train", None) and len(train_df) > args.max_train:
        train_df = train_df.sample(n=args.max_train, random_state=args.seed)
        print(f"Capped training set to {args.max_train} examples")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = DatasetClass(train_df, tokenizer, max_length=max_len)
    val_dataset   = DatasetClass(val_df,   tokenizer, max_length=max_len)
    test_dataset  = DatasetClass(test_df,  tokenizer, max_length=max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
    )
    model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Training loop
    best_val_acc = 0
    model_slug = args.model_name.split("/")[-1].lower()
    output_name = f"{args.mode}_{model_slug}"
    if args.test_scenario is not None:
        output_name += f"_scenario{args.test_scenario}"
    model_dir = os.path.join(OUTPUT_DIR, output_name)
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, val_report, _, _ = evaluate(model, val_loader, device)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
            print(f"  → Saved best model (val_acc={val_acc:.4f})")

    # Final evaluation on test set
    print("\n" + "=" * 50)
    print("Final Test Evaluation")
    print("=" * 50)

    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt"), weights_only=True))
    test_loss, test_acc, test_report, test_preds, test_labels = evaluate(model, test_loader, device)

    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nPer-class results:")
    for label in LABEL_NAMES:
        if label in test_report:
            r = test_report[label]
            print(f"  {label:20s}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f}")

    # Save results
    results = {
        "mode": args.mode,
        "model_name": args.model_name,
        "model_slug": model_slug,
        "test_scenario": args.test_scenario,
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
        "per_class": test_report,
        "args": vars(args),
    }
    with open(os.path.join(model_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {model_dir}/results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT on Circa")
    parser.add_argument("--mode", type=str,
                        choices=["baseline", "repeat", "template", "oracle", "llm"],
                        default="baseline", help="Training mode")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace model checkpoint")
    parser.add_argument("--max_train", type=int, default=None,
                        help="Cap training set size (None = use all)")
    parser.add_argument("--test_scenario", type=int, default=None,
                        help="Scenario index for leave-one-out (0-9). None = matched split.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_training(args)
