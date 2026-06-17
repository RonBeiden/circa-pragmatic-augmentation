"""
Generate LLM chain-of-thought explanations for Circa QA pairs.
Supports Azure OpenAI (primary) and Gemini (fallback).
Outputs: data/circa_with_explanations.csv
"""

import os
import time
import argparse
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INPUT_FILE = os.path.join(DATA_DIR, "circa_processed.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "circa_with_explanations.csv")

# Number of examples to generate explanations for
MAX_EXAMPLES = 5000

SYSTEM_PROMPT = """You are an expert in pragmatics and conversational implicature. 
Given a yes/no question and an indirect answer (one that doesn't explicitly say "yes" or "no"), 
provide a brief chain-of-thought explanation (2-3 sentences) of what the answer pragmatically implies.

Focus on:
1. What the literal content of the answer tells us
2. What can be inferred from the answer given the conversational context
3. Whether this implies yes, no, uncertainty, or a condition

Be concise and analytical. Do NOT state the final label - just explain the reasoning."""

USER_TEMPLATE = """Context: {context}
Question: {question}
Answer: {answer}

Explain what this answer pragmatically implies about the yes/no question:"""


# ─── LLM Provider Classes ────────────────────────────────────────────────────

class AzureOpenAIProvider:
    """Azure OpenAI provider (HP corporate) — uses gen_llm from functions.py.
    Handles token refresh on timeout/auth errors."""

    MAX_RETRIES = 3
    RETRY_DELAY = 10  # seconds between retries

    def __init__(self):
        import sys
        # Add project root to path so we can import functions/settings
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        self._init_llm()

    def _init_llm(self):
        """Initialize or refresh the LLM connection."""
        from functions import gen_llm
        self.access_token, self.llm = gen_llm("gpt-4.1-mini")
        print(f"Using Azure OpenAI via gen_llm — deployment: gpt-4.1-mini")

    def _refresh(self):
        """Refresh token and reinitialize LLM on auth/timeout failure."""
        print("\n⟳ Refreshing Azure OpenAI connection (token may have expired)...")
        time.sleep(5)
        self._init_llm()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.llm.invoke(messages)
                return response.content.strip()
            except Exception as e:
                err_str = str(e).lower()
                is_retriable = any(kw in err_str for kw in [
                    "timeout", "timed out", "401", "unauthorized",
                    "token", "expired", "connection", "429", "rate"
                ])
                if is_retriable and attempt < self.MAX_RETRIES - 1:
                    print(f"\n⚠ Attempt {attempt+1} failed: {e}")
                    self._refresh()
                    time.sleep(self.RETRY_DELAY)
                else:
                    raise

    @property
    def rate_limit_batch(self):
        return 50  # Azure has generous rate limits

    @property
    def rate_limit_pause(self):
        return 1


class GeminiProvider:
    """Google Gemini provider."""

    def __init__(self):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash"
        print(f"Using Gemini — model: {self.model}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=f"{system_prompt}\n\n{user_prompt}",
        )
        return response.text.strip()

    @property
    def rate_limit_batch(self):
        return 14  # Free tier: ~15 RPM

    @property
    def rate_limit_pause(self):
        return 4


def get_provider(name: str = None):
    """Get LLM provider by name. Defaults to LLM_PROVIDER env var."""
    provider_name = name or os.getenv("LLM_PROVIDER", "azure")
    if provider_name == "azure":
        return AzureOpenAIProvider()
    elif provider_name == "gemini":
        return GeminiProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Use 'azure' or 'gemini'.")


# ─── Explanation Generation ──────────────────────────────────────────────────

def generate_explanations(df: pd.DataFrame, max_examples: int = MAX_EXAMPLES,
                          provider_name: str = None, resume: bool = False) -> pd.DataFrame:
    """Generate explanations for a subset of examples. Supports resuming from checkpoint."""
    provider = get_provider(provider_name)

    # Sample stratified by label to maintain distribution
    if len(df) > max_examples:
        df_sample = df.groupby("label_6class", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max_examples // 6), random_state=42)
        ).reset_index(drop=True)
        # Fill remaining slots randomly
        remaining = max_examples - len(df_sample)
        if remaining > 0:
            extra_ids = df[~df["id"].isin(df_sample["id"])].sample(n=remaining, random_state=42)
            df_sample = pd.concat([df_sample, extra_ids]).reset_index(drop=True)
    else:
        df_sample = df.copy()

    # Resume: load existing checkpoint and figure out where we left off
    start_idx = 0
    if resume and os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        # Match by id to find which rows already have explanations
        if "explanation" in existing.columns:
            done = existing[existing["explanation"].notna() & (existing["explanation"] != "")]
            done_ids = set(done["id"].tolist())
            # Ensure df_sample uses the same rows as previous run
            if len(existing) == len(df_sample):
                df_sample = existing.copy()
                # Count how many are already done
                start_idx = len(done)
                print(f"\n↻ Resuming from checkpoint: {start_idx}/{len(df_sample)} already done")
            else:
                print(f"\n⚠ Checkpoint has {len(existing)} rows but expected {len(df_sample)}. Starting fresh.")
    
    print(f"Generating explanations for {len(df_sample)} examples (starting at {start_idx})...")

    # Pre-populate explanations list from existing data
    if "explanation" in df_sample.columns and start_idx > 0:
        explanations = df_sample["explanation"].tolist()[:start_idx]
    else:
        explanations = []
        if "explanation" not in df_sample.columns:
            df_sample["explanation"] = ""

    for idx in tqdm(range(start_idx, len(df_sample)), initial=start_idx, total=len(df_sample)):
        row = df_sample.iloc[idx]
        prompt = USER_TEMPLATE.format(
            context=row["context"],
            question=row["question"],
            answer=row["answer"]
        )

        try:
            explanation = provider.generate(SYSTEM_PROMPT, prompt)
        except Exception as e:
            print(f"\nError on row {row['id']}: {e}")
            explanation = ""
            time.sleep(5)

        explanations.append(explanation)

        # Rate limiting
        if (idx + 1) % provider.rate_limit_batch == 0:
            time.sleep(provider.rate_limit_pause)

        # Checkpoint: save progress every 50 examples
        if (idx + 1) % 50 == 0:
            df_sample.loc[df_sample.index[:len(explanations)], "explanation"] = explanations
            df_sample.to_csv(OUTPUT_FILE, index=False)
            print(f"\n  ✓ Checkpoint saved ({idx+1}/{len(df_sample)})")

    df_sample["explanation"] = explanations

    # Final save
    df_sample.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df_sample)} examples with explanations to {OUTPUT_FILE}")

    # Print some examples
    print("\n--- Sample explanations ---")
    for _, row in df_sample.head(3).iterrows():
        print(f"\nQ: {row['question']}")
        print(f"A: {row['answer']}")
        print(f"Label: {row['goldstandard1']}")
        print(f"Explanation: {row['explanation']}")

    return df_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLM explanations for Circa")
    parser.add_argument("--provider", type=str, choices=["azure", "gemini"],
                        default=None, help="LLM provider (default: from .env LLM_PROVIDER)")
    parser.add_argument("--max_examples", type=int, default=MAX_EXAMPLES,
                        help="Max examples to generate explanations for")
    parser.add_argument("--test", action="store_true",
                        help="Quick test: generate 1 explanation and exit")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint if output file exists")
    args = parser.parse_args()

    if args.test:
        # Quick hello test
        print("Running quick test...")
        provider = get_provider(args.provider)
        result = provider.generate(
            "You are a helpful assistant.",
            "Say hello in one sentence."
        )
        print(f"Test response: {result}")
        print("\n✓ Provider is working!")
        exit(0)

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run download_data.py first.")
        exit(1)

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} examples from {INPUT_FILE}")

    generate_explanations(df, max_examples=args.max_examples, provider_name=args.provider,
                           resume=args.resume)
