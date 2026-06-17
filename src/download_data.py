"""
Download and preprocess the Circa dataset.
Outputs: data/circa_processed.csv with columns:
    id, context, question, answer, canquestion, goldstandard1, goldstandard2, scenario_idx
"""

import os
import urllib.request
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_FILE = os.path.join(DATA_DIR, "circa-data.tsv")
PROCESSED_FILE = os.path.join(DATA_DIR, "circa_processed.csv")

CIRCA_URL = "https://raw.githubusercontent.com/google-research-datasets/circa/main/circa-data.tsv"

# The 10 social situations (0-indexed)
SCENARIOS = [
    "X wants to know about Y's food preferences",
    "X wants to know what activities Y likes to do during weekends",
    "X wants to know what sorts of books Y likes to read",
    "Y has just moved into a neighbourhood and meets his/her new neighbour X",
    "X and Y are colleagues who are leaving work on a Friday at the same time",
    "X wants to know about Y's music preferences",
    "Y has just travelled from a different city to meet X",
    "X and Y are childhood neighbours who unexpectedly run into each other at a cafe",
    "Y has just told X that he/she is thinking of buying a flat in New York",
    "Y has just told X that he/she is considering switching his/her job",
]

# Label mapping for the strict 6-class setting
LABEL_MAP_6CLASS = {
    "Yes": 0,
    "Probably yes / sometimes yes": 1,
    "Yes, subject to some conditions": 2,
    "No": 3,
    "Probably no": 4,
    "In the middle, neither yes nor no": 5,
}

# Label mapping for relaxed 4-class setting
LABEL_MAP_4CLASS = {
    "Yes": 0,
    "Probably yes / sometimes yes": 0,
    "Yes, subject to some conditions": 1,
    "No": 2,
    "Probably no": 2,
    "In the middle, neither yes nor no": 3,
}


def download_circa():
    """Download the raw Circa TSV file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(RAW_FILE):
        print(f"Downloading Circa dataset to {RAW_FILE}...")
        urllib.request.urlretrieve(CIRCA_URL, RAW_FILE)
        print("Done.")
    else:
        print(f"Circa dataset already exists at {RAW_FILE}")


def process_circa():
    """Process raw TSV into a clean CSV with labels."""
    print("Processing Circa dataset...")

    df = pd.read_csv(RAW_FILE, sep="\t", header=0,
                     names=["id", "context", "question", "canquestion", "answer",
                            "judgements", "goldstandard1", "goldstandard2"])

    # Map context to scenario index
    def get_scenario_idx(context):
        for i, scenario in enumerate(SCENARIOS):
            if scenario.lower() in context.lower():
                return i
        return -1

    df["scenario_idx"] = df["context"].apply(get_scenario_idx)

    # Filter to rows with valid gold standard labels (strict 6-class)
    valid_labels = set(LABEL_MAP_6CLASS.keys())
    df_valid = df[df["goldstandard1"].isin(valid_labels)].copy()

    # Add numeric labels
    df_valid["label_6class"] = df_valid["goldstandard1"].map(LABEL_MAP_6CLASS)
    df_valid["label_4class"] = df_valid["goldstandard1"].map(LABEL_MAP_4CLASS)

    # Keep relevant columns
    df_valid = df_valid[["id", "context", "question", "canquestion", "answer",
                         "goldstandard1", "label_6class", "label_4class", "scenario_idx"]]

    df_valid.to_csv(PROCESSED_FILE, index=False)
    print(f"Processed {len(df_valid)} examples (from {len(df)} total)")
    print(f"Saved to {PROCESSED_FILE}")

    # Print label distribution
    print("\n6-class distribution:")
    print(df_valid["goldstandard1"].value_counts())
    print(f"\nScenario distribution:")
    print(df_valid["scenario_idx"].value_counts().sort_index())

    return df_valid


if __name__ == "__main__":
    download_circa()
    process_circa()
