# Pragmatic Reasoning Augmentation for Indirect Answer Classification

## Research Question
Can enriching BERT fine-tuning with LLM-generated chain-of-thought explanations improve indirect answer classification on the Circa dataset?

## Project Structure
```
circa-pragmatic-augmentation/
├── data/               # Downloaded Circa data + generated explanations
├── src/
│   ├── download_data.py        # Download and preprocess Circa
│   ├── generate_explanations.py # Generate LLM explanations
│   ├── dataset.py              # PyTorch dataset classes
│   ├── train.py                # BERT fine-tuning (baseline + augmented)
│   └── evaluate.py             # Evaluation + leave-one-out
├── outputs/            # Model checkpoints, results
├── notebooks/          # Analysis notebooks
├── requirements.txt
└── .env                # API keys (not committed)
```

## Setup
```bash
pip install -r requirements.txt
```

Create a `.env` file with your API key:
```
OPENAI_API_KEY=your-key-here
```

## Pipeline
1. `python src/download_data.py` — Download Circa dataset
2. `python src/generate_explanations.py` — Generate LLM explanations for training subset
3. `python src/train.py --mode baseline` — Train BERT-MNLI-YN baseline
4. `python src/train.py --mode augmented` — Train with explanations
5. `python src/evaluate.py` — Compare models (matched + leave-one-out)

## Reference
Louis, A., Roth, D., & Radlinski, F. (2020). "I'd rather just go to bed": Understanding Indirect Answers. EMNLP 2020.
