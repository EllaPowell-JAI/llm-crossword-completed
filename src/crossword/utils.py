"""Utility functions for loading and saving crossword puzzles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.crossword.crossword import CrosswordPuzzle
from src.crossword.exceptions import InvalidClueError
from src.crossword.types import Clue, Direction


def load_puzzle(file_path: str | Path) -> CrosswordPuzzle:
    """Load a puzzle from a JSON file, or a .puz file when puzpy is installed."""
    path = Path(file_path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return CrosswordPuzzle.model_validate(json.load(f))

    if path.suffix.lower() == ".puz":
        try:
            import puz  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("Install puzpy to load .puz files") from exc
        return _load_puz(path, puz)

    raise ValueError(f"Unsupported puzzle format: {path.suffix}")


def save_puzzle(puzzle: CrosswordPuzzle, file_path: str | Path) -> None:
    """Serialize a puzzle to JSON."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(puzzle.model_dump(mode="json", by_alias=True), f, indent=2)
        f.write("\n")


def clone_puzzle_without_answers(puzzle: CrosswordPuzzle) -> CrosswordPuzzle:
    """Return a copy with answer fields removed.

    This is useful for tests that ensure the solver does not inspect the stored
    answer fields while still allowing the original puzzle to be used for
    validation afterwards.
    """
    data: dict[str, Any] = puzzle.model_dump(mode="json", by_alias=True)
    for clue in data.get("clues", []):
        clue["answer"] = None
        clue["answered"] = False
    data["grid_history"] = []
    data["clue_history"] = []
    return CrosswordPuzzle.model_validate(data)


def _load_puz(path: Path, puz_module: Any) -> CrosswordPuzzle:  # pragma: no cover - optional path
    puzzle_file = puz_module.read(str(path))
    puzzle = CrosswordPuzzle(width=puzzle_file.width, height=puzzle_file.height)
    numbering = puzzle_file.clue_numbering()

    for across in numbering.across:
        cell = across["cell"]
        answer = "".join(puzzle_file.solution[cell + i] for i in range(across["len"]))
        try:
            puzzle.add_clue(
                Clue(
                    number=across["num"],
                    text=across["clue"],
                    direction=Direction.ACROSS,
                    length=across["len"],
                    row=cell // puzzle.width,
                    col=cell % puzzle.width,
                    answer=answer,
                )
            )
        except InvalidClueError as exc:
            raise InvalidClueError(f"Invalid across clue: {exc}") from exc

    for down in numbering.down:
        cell = down["cell"]
        answer = "".join(puzzle_file.solution[cell + i * puzzle.width] for i in range(down["len"]))
        try:
            puzzle.add_clue(
                Clue(
                    number=down["num"],
                    text=down["clue"],
                    direction=Direction.DOWN,
                    length=down["len"],
                    row=cell // puzzle.width,
                    col=cell % puzzle.width,
                    answer=answer,
                )
            )
        except InvalidClueError as exc:
            raise InvalidClueError(f"Invalid down clue: {exc}") from exc

    return puzzle
