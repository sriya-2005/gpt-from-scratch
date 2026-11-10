# GPT from scratch

A GPT-2-style language model (12 layers, 12 heads, 768 dimensions, ~124M parameters) written
from scratch in PyTorch, trained on a laptop CPU, plus an experiment in teaching it to answer
questions about Edith Wharton's short story "The Verdict" using only its own weights (no
retrieval, no pretrained model).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\check_env.py
```

## Pipeline

| Step | Command |
|---|---|
| Prepare data | `python scripts/prepare_data.py --dataset verdict` (also `verdict_full`, `verdict_facts`, `wikipedia`) |
| Train | `python scripts/train.py --preset verdict` (presets: `verdict`, `verdict_story`, `verdict_facts`, `small`, `gpt2`) |
| Generate | `python scripts/inference.py --ckpt <checkpoint> --prompt "Every effort moves you"` |
| Desktop app | `python scripts/gui.py <checkpoint>` |

## Question-answering experiment

| Step | Script |
|---|---|
| Story-knowledge test | `scripts/eval_story.py` |
| Build / check the QA dataset | `scripts/build_qa_dataset.py` (source: `data/qa/verdict_qa.jsonl`) |
| QA fine-tuning | `scripts/finetune_qa.py` |
| QA evaluation | `scripts/eval_qa.py --ckpt <checkpoint> --split val` |

### Open book: answering from retrieved passages, with conversation memory

The closed-book model barely answers new wordings, so a second mode looks the answer up first.
The story is cut into 130-token passages and each question is matched to them with BM25
keyword search (`scripts/retrieval.py`, written from scratch; the hand-written fact statements
are indexed with the passages they describe). The model reads the best passage and answers
from it, moving on to the 2nd and 3rd passage if it says the passage does not contain the
answer. Earlier questions are part of the search, so follow-ups such as "How did he die?"
after "Who was Stroud?" still find the right passage.

| Step | Command |
|---|---|
| Search quality (single questions and follow-ups) | `python scripts/eval_retrieval.py` |
| Fine-tune with passages | `python scripts/finetune_qa.py --with-context --init-from models/checkpoints/verdict_facts/best.pt --rehearsal-data verdict_facts --run-name verdict_rag` |
| Evaluate | `python scripts/eval_qa.py --ckpt models/checkpoints/verdict_rag/best.pt --split val` |
| Chat | `python scripts/chat.py --ckpt models/checkpoints/verdict_rag/best.pt` (`--search-only` without a model) |

`data/qa/followups.jsonl` holds ten multi-turn conversations used to test the follow-up search.

`data/qa/` holds the hand-checked questions (every answer cites an exact quote from the
story), the reworded fact statements, the frozen train/val/test split and the held-out
probe questions. Results are in `logs/` and `models/checkpoints/*/log.csv`.

Model weights (`*.pt`) are not in the repository because of their size.
