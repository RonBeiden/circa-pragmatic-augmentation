"""
Aggregate trained results into comparison tables and orchestrate training.

Usage:
  python src/evaluate.py                            # Print summary of all trained models
  python src/evaluate.py --action train_all         # Train all 5 modes x 3 models (15 runs)
  python src/evaluate.py --action train_all --max_train 5000  # Faster with capped data
  python src/evaluate.py --action train_loo         # LOO: baseline+llm x 3 models x 10 folds
  python src/evaluate.py --action train_all --mode llm --model_name roberta-base  # Single run
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

MODELS = [
    "bert-base-uncased",
    "textattack/bert-base-uncased-MNLI",
    "roberta-base",
]
MODES = ["baseline", "repeat", "template", "oracle", "llm"]


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def collect_results() -> pd.DataFrame:
    """Scan outputs/ for results.json files and return an aggregated DataFrame."""
    rows = []
    if not os.path.exists(OUTPUT_DIR):
        return pd.DataFrame()
    for subdir in sorted(os.listdir(OUTPUT_DIR)):
        results_path = os.path.join(OUTPUT_DIR, subdir, "results.json")
        if not os.path.exists(results_path):
            continue
        with open(results_path) as f:
            r = json.load(f)
        stored_slug = r.get("model_slug")
        if not stored_slug:
            mn = r.get("model_name") or r.get("args", {}).get("model_name", "unknown")
            stored_slug = model_slug(mn)
        pc = r.get("per_class", {})
        label_names = ["Yes", "Probably yes", "Yes, conditional", "No", "Probably no", "In the middle"]
        macro_f1 = float(np.mean([pc.get(l, {}).get("f1-score", 0.0) for l in label_names]))
        rows.append({
            "mode":      r.get("mode", "?"),
            "model":     stored_slug,
            "scenario":  r.get("test_scenario"),
            "test_acc":  r.get("test_accuracy"),
            "val_acc":   r.get("best_val_accuracy"),
            "macro_f1":  macro_f1,
        })
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Matched split table
    matched = df[df["scenario"].isna()].copy()
    if not matched.empty:
        for metric, col in [("Test Accuracy", "test_acc"), ("Macro-F1", "macro_f1")]:
            pivot = matched.pivot_table(index="mode", columns="model",
                                        values=col, aggfunc="first")
            pivot = pivot.reindex([m for m in MODES if m in pivot.index])
            print("\n" + "=" * 70)
            print(f"MATCHED SPLIT -- {metric}")
            print("=" * 70)
            print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))
            pivot.to_csv(os.path.join(OUTPUT_DIR, f"results_matched_{col}.csv"))

    # Leave-one-out table
    loo = df[df["scenario"].notna()].copy()
    if not loo.empty:
        loo["scenario"] = pd.to_numeric(loo["scenario"], errors="coerce")
        loo = loo.dropna(subset=["scenario"])
        if not loo.empty:
            loo_mean = loo.groupby(["mode", "model"])["test_acc"].mean().reset_index()
            pivot_loo = loo_mean.pivot(index="mode", columns="model", values="test_acc")
            pivot_loo = pivot_loo.reindex([m for m in MODES if m in pivot_loo.index])
            print("\n" + "=" * 70)
            print("LEAVE-ONE-OUT -- Mean Test Accuracy (10 scenarios)")
            print("=" * 70)
            print(pivot_loo.to_string(float_format=lambda x: f"{x:.4f}"))
            pivot_loo.to_csv(os.path.join(OUTPUT_DIR, "results_loo.csv"))

    df.to_csv(os.path.join(OUTPUT_DIR, "all_results.csv"), index=False)
    print(f"\nFull results saved to {OUTPUT_DIR}")


def run_all_training(args):
    """Train all mode x model combinations for matched split (skips already done)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train import run_training
    import torch

    modes_to_run  = ([args.mode]       if (getattr(args, "mode", None)       and args.mode       in MODES)
                     else MODES)
    models_to_run = ([args.model_name] if (getattr(args, "model_name", None) and args.model_name in MODELS)
                     else MODELS)

    for model_name in models_to_run:
        for mode in modes_to_run:
            slug = model_slug(model_name)
            out_dir = os.path.join(OUTPUT_DIR, f"{mode}_{slug}")
            if os.path.exists(os.path.join(out_dir, "results.json")):
                print(f"[skip] {mode} / {slug} (already done)")
                continue
            print(f"\n{'='*60}")
            print(f"Training  mode={mode}  model={model_name}")
            print(f"{'='*60}")
            args.mode          = mode
            args.model_name    = model_name
            args.test_scenario = None
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            run_training(args)


def run_loo_training(args):
    """LOO training for baseline+llm across all 3 models x 10 scenarios."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train import run_training
    import torch

    loo_modes  = ([args.mode]       if (getattr(args, "mode", None)       and args.mode       in MODES)
                  else ["baseline", "llm"])
    loo_models = ([args.model_name] if (getattr(args, "model_name", None) and args.model_name in MODELS)
                  else MODELS)

    for model_name in loo_models:
        for mode in loo_modes:
            for scenario in range(10):
                slug = model_slug(model_name)
                out_dir = os.path.join(OUTPUT_DIR, f"{mode}_{slug}_scenario{scenario}")
                if os.path.exists(os.path.join(out_dir, "results.json")):
                    print(f"[skip] {mode}/{slug}/s{scenario}")
                    continue
                print(f"\n{'='*60}")
                print(f"LOO  mode={mode}  model={model_name}  scenario={scenario}")
                print(f"{'='*60}")
                args.mode          = mode
                args.model_name    = model_name
                args.test_scenario = scenario
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                run_training(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate or train all Circa conditions")
    parser.add_argument("--action",     choices=["evaluate", "train_all", "train_loo"],
                        default="evaluate")
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--epochs",     type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--max_train",  type=int,   default=None,
                        help="Cap training size per run (None = use all data)")
    parser.add_argument("--model_name", type=str,   default=None,
                        help="Restrict to one model (default: all 3)")
    parser.add_argument("--mode",       type=str,   default=None,
                        help="Restrict to one mode (default: all 5 or baseline+llm for LOO)")
    args = parser.parse_args()

    if args.action == "evaluate":
        df = collect_results()
        if df.empty:
            print("No results found in outputs/. Train models first with --action train_all")
        else:
            print(f"Found {len(df)} trained model results.")
            print_summary(df)
    elif args.action == "train_all":
        run_all_training(args)
    elif args.action == "train_loo":
        run_loo_training(args)