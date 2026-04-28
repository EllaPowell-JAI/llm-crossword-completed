from pathlib import Path

import pytest

from src.crossword.solver import PuzzleSolver, StaticKnowledgeAnswerProvider
from src.crossword.utils import clone_puzzle_without_answers, load_puzzle


@pytest.mark.parametrize("name", ["easy", "medium", "hard", "cryptic"])
def test_solver_completes_bundled_puzzles_without_reading_answer_fields(name):
    original = load_puzzle(Path("data") / f"{name}.json")
    hidden_answers_puzzle = clone_puzzle_without_answers(original)

    result = PuzzleSolver(StaticKnowledgeAnswerProvider()).solve(hidden_answers_puzzle)

    assert result.success, result.diagnostics
    expected = {clue.label: clue.answer for clue in original.clues}
    assert result.answers == expected


def test_solver_fills_grid_and_validates_easy():
    puzzle = load_puzzle("data/easy.json")
    result = PuzzleSolver().solve(puzzle)
    assert result.success
    assert result.grid == ["CAT##", "O#E##", "W#A##", "##R##", "#####"]
    assert puzzle.validate_all()
