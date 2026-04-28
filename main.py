"""Command line entry point for the completed crossword solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from src.crossword.solver import PuzzleSolver, build_provider
from src.crossword.utils import load_puzzle

DEFAULT_PUZZLES = [
    "data/easy.json",
    "data/medium.json",
    "data/hard.json",
    "data/cryptic.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve the bundled crossword puzzles.")
    parser.add_argument("puzzles", nargs="*", default=DEFAULT_PUZZLES, help="Puzzle JSON files to solve")
    parser.add_argument(
        "--provider",
        choices=["static", "azure", "auto"],
        default="static",
        help="Candidate generator. 'static' works offline; 'azure' uses Azure OpenAI from .env.",
    )
    parser.add_argument("--write-solutions", action="store_true", help="Write JSON outputs to solutions/")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    provider = build_provider(args.provider)
    solver = PuzzleSolver(answer_provider=provider)
    all_ok = True

    for puzzle_path in args.puzzles:
        puzzle = load_puzzle(puzzle_path)
        result = solver.solve(puzzle)
        all_ok = all_ok and result.success

        print(f"\n=== {puzzle_path} ===")
        print("success:", result.success)
        print("grid:")
        print("\n".join(result.grid))
        print("answers:")
        for clue_id, answer in result.answers.items():
            print(f"  {clue_id}: {answer}")
        if puzzle.validate_all():
            print("validated against bundled answers: yes")
        elif result.success:
            print("validated against bundled answers: no")
        for message in result.diagnostics:
            print("diagnostic:", message)

        if args.write_solutions:
            out_dir = Path("solutions")
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"{Path(puzzle_path).stem}_solution.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result.as_dict(), f, indent=2)
                f.write("\n")
            print(f"wrote: {out_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
