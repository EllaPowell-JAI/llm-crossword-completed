import pytest

from src.crossword.crossword import CrosswordPuzzle
from src.crossword.exceptions import InvalidGridError
from src.crossword.types import Clue, Direction


def test_grid_fill_undo_and_validation():
    puzzle = CrosswordPuzzle(
        width=5,
        height=5,
        clues=[
            Clue(number=1, text="Feline friend", direction=Direction.ACROSS, length=3, row=0, col=0, answer="CAT"),
            Clue(number=1, text="Dairy farm animal", direction=Direction.DOWN, length=3, row=0, col=0, answer="COW"),
            Clue(number=2, text="A drop of sadness", direction=Direction.DOWN, length=4, row=0, col=2, answer="TEAR"),
        ],
    )

    puzzle.set_clue_chars(puzzle.clues[0], list("CAT"))
    puzzle.set_clue_chars(puzzle.clues[1], list("COW"))
    assert not puzzle.validate_all()
    puzzle.set_clue_chars(puzzle.clues[2], list("TEAR"))
    assert puzzle.validate_all()
    assert puzzle.to_lines() == ["CAT##", "O#E##", "W#A##", "##R##", "#####"]

    puzzle.undo()
    assert puzzle.answer_for(puzzle.clues[2]) is None
    puzzle.reset()
    assert all(not clue.answered for clue in puzzle.clues)


def test_conflicting_crossing_letters_are_rejected():
    puzzle = CrosswordPuzzle(
        width=3,
        height=3,
        clues=[
            Clue(number=1, text="across", direction=Direction.ACROSS, length=3, row=0, col=0, answer="CAT"),
            Clue(number=2, text="down", direction=Direction.DOWN, length=3, row=0, col=0, answer="DOG"),
        ],
    )
    puzzle.set_clue_chars(puzzle.clues[0], list("CAT"))
    with pytest.raises(InvalidGridError):
        puzzle.set_clue_chars(puzzle.clues[1], list("DOG"))
