# LLM Crossword Solver Challenge - Completed

This repository contains a completed solution for the `llm-crossword` interview challenge.

The implementation has two stages:

1. **Candidate generation**: an answer provider proposes clue answers. The default provider is deterministic and offline for the bundled challenge puzzles. An Azure OpenAI provider is also included for live LLM-backed candidate generation.
2. **Crossing resolution**: a constraint-search solver chooses the globally consistent answer assignment by enforcing all across/down intersections.

The solver does **not** read the puzzle JSON `answer` fields during solving. The included answers remain in `data/*.json` only so the tests and CLI can validate the final grid.

## What was answered

The repository now solves all supplied puzzles:

- `data/easy.json`
- `data/medium.json`
- `data/hard.json`
- `data/cryptic.json`

The completed clue answers and grids are in `ANSWERS.md` and in machine-readable form under `solutions/`.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill .env with your Azure OpenAI values when using --provider azure.
```

The `.env` file is intentionally ignored by git.

## Run

Offline deterministic mode:

```bash
python main.py
```

Write solution JSON files:

```bash
python main.py --write-solutions
```

Azure OpenAI mode:

```bash
python main.py --provider azure
```

Automatic mode uses the offline knowledge base and also adds Azure candidates when the Azure environment variables are present:

```bash
python main.py --provider auto
```

## Test

```bash
pytest
```

Expected result in this completed repo:

```text
7 passed
```

## Key files

```text
src/crossword/crossword.py  # puzzle state, validation, grid fill/undo/reset
src/crossword/solver.py     # answer providers and CSP crossing solver
src/crossword/utils.py      # load/save helpers
data/*.json                 # challenge puzzles
solutions/*.json            # generated solution outputs
ANSWERS.md                  # human-readable solved grids and answers
```

## Security note

`.env` is for local secrets only and is listed in `.gitignore`. Do not commit API keys to the repository history or publish them in a remote GitHub repo.
