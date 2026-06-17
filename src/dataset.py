"""
PyTorch Dataset classes for Circa with and without explanations.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class CircaDataset(Dataset):
    """Dataset for baseline BERT-MNLI-YN (question + answer only)."""

    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = 128,
                 num_classes: int = 6):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes
        self.label_col = "label_6class" if num_classes == 6 else "label_4class"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = str(row["question"])
        answer = str(row["answer"])

        encoding = self.tokenizer(
            question, answer,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        tti = encoding.get("token_type_ids")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": tti.squeeze(0) if tti is not None else torch.zeros(self.max_length, dtype=torch.long),
            "labels": torch.tensor(row[self.label_col], dtype=torch.long),
        }


class CircaAugmentedDataset(Dataset):
    """Dataset for augmented model (question + answer + explanation)."""

    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = 256,
                 num_classes: int = 6):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes
        self.label_col = "label_6class" if num_classes == 6 else "label_4class"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = str(row["question"])
        answer = str(row["answer"])
        explanation = str(row.get("explanation", ""))

        # Format: [CLS] question [SEP] answer. Explanation: <explanation> [SEP]
        text_a = question
        text_b = f"{answer}. Explanation: {explanation}" if explanation else answer

        encoding = self.tokenizer(
            text_a, text_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        tti = encoding.get("token_type_ids")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": tti.squeeze(0) if tti is not None else torch.zeros(self.max_length, dtype=torch.long),
            "labels": torch.tensor(row[self.label_col], dtype=torch.long),
        }


def create_splits(df: pd.DataFrame, test_scenario: int = None, val_ratio: float = 0.1,
                  random_state: int = 42):
    """
    Create train/val/test splits.

    If test_scenario is provided: leave-one-out split (test = that scenario, train = rest).
    Otherwise: random 80/10/10 split (matched setting).
    """
    # Drop rows with missing labels and coerce to int (handles NaN from CSV merge)
    for col in ["label_6class", "label_4class"]:
        if col in df.columns:
            df = df.dropna(subset=[col])
            df = df.copy()
            df[col] = df[col].astype(int)

    if test_scenario is not None:
        # Leave-one-out: test on specified scenario
        test_df = df[df["scenario_idx"] == test_scenario]
        train_val_df = df[df["scenario_idx"] != test_scenario]
    else:
        # Random split (matched)
        test_df = df.sample(frac=0.1, random_state=random_state)
        train_val_df = df.drop(test_df.index)

    # Split train_val into train and val
    val_df = train_val_df.sample(frac=val_ratio, random_state=random_state)
    train_df = train_val_df.drop(val_df.index)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Constants + helpers for alternative augmentation conditions
# ---------------------------------------------------------------------------

LABEL_DESCRIPTIONS = [
    "The answer clearly expresses agreement.",
    "The answer probably expresses agreement.",
    "The answer expresses conditional agreement.",
    "The answer clearly expresses disagreement.",
    "The answer probably expresses disagreement.",
    "The answer is ambiguous or evasive.",
]


def get_template_hint(answer: str) -> str:
    """Rule-based surface hint from answer keywords (no gold label used)."""
    a = answer.lower()
    if any(w in a for w in ["yes", "yeah", "sure", "absolutely", "definitely",
                             "of course", "certainly", "love to", "would love", "enjoy"]):
        return "The answer contains affirmative language."
    if any(w in a for w in ["no", "not ", "never", "nope", "can't", "cannot",
                             "don't", "won't", "wouldn't", "afraid not"]):
        return "The answer contains negative language."
    if any(w in a for w in ["maybe", "perhaps", "probably", "might", "could",
                             "possibly", "sometimes", "kind of", "sort of"]):
        return "The answer expresses uncertainty."
    if any(w in a for w in ["if ", "unless", "depends", "as long", "provided",
                             "only if", "when ", "conditional"]):
        return "The answer contains conditional language."
    return "The answer is indirect or ambiguous."


class CircaRepeatDataset(Dataset):
    """Control: Q + A + A. Tests whether extra sequence length (not content) drives gain."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 192, num_classes: int = 6):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes
        self.label_col = "label_6class" if num_classes == 6 else "label_4class"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = str(row["question"])
        answer = str(row["answer"])
        encoding = self.tokenizer(
            question, f"{answer} {answer}",
            max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        tti = encoding.get("token_type_ids")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": tti.squeeze(0) if tti is not None else torch.zeros(self.max_length, dtype=torch.long),
            "labels": torch.tensor(row[self.label_col], dtype=torch.long),
        }


class CircaTemplateDataset(Dataset):
    """Control: Q + A + rule-based keyword hint. Tests keyword heuristic vs. LLM reasoning."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 192, num_classes: int = 6):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes
        self.label_col = "label_6class" if num_classes == 6 else "label_4class"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = str(row["question"])
        answer = str(row["answer"])
        hint = get_template_hint(answer)
        encoding = self.tokenizer(
            question, f"{answer}. Hint: {hint}",
            max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        tti = encoding.get("token_type_ids")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": tti.squeeze(0) if tti is not None else torch.zeros(self.max_length, dtype=torch.long),
            "labels": torch.tensor(row[self.label_col], dtype=torch.long),
        }


class CircaOracleDataset(Dataset):
    """Upper bound: Q + A + gold label description. Uses true label -- marks performance ceiling."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 192, num_classes: int = 6):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes
        self.label_col = "label_6class" if num_classes == 6 else "label_4class"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = str(row["question"])
        answer = str(row["answer"])
        label_idx = int(row[self.label_col])
        desc = LABEL_DESCRIPTIONS[label_idx]
        encoding = self.tokenizer(
            question, f"{answer}. Note: {desc}",
            max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        tti = encoding.get("token_type_ids")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": tti.squeeze(0) if tti is not None else torch.zeros(self.max_length, dtype=torch.long),
            "labels": torch.tensor(row[self.label_col], dtype=torch.long),
        }
