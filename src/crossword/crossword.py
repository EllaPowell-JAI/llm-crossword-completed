"""Crossword puzzle state and grid manipulation."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .exceptions import InvalidClueError, InvalidGridError
from .types import Cell, Clue, Direction, Grid


class CrosswordPuzzle(BaseModel):
    """A crossword puzzle with undoable grid state.

    The challenge input stores all cells as blank and provides clue metadata. The
    solver fills slots by calling ``set_clue_chars``; every call appends a new grid
    snapshot, so the original interview helper methods still work.
    """

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    clues: List[Clue] = Field(default_factory=list)
    grid_history: List[Grid] = Field(default_factory=list, alias="grid_history")
    clue_history: List[int] = Field(default_factory=list, alias="clue_history")

    @model_validator(mode="after")
    def ensure_grid(self) -> "CrosswordPuzzle":
        if not self.grid_history:
            self.grid_history.append(Grid.empty(width=self.width, height=self.height))
        for grid in self.grid_history:
            if grid.width != self.width or grid.height != self.height:
                raise InvalidGridError("grid dimensions do not match puzzle dimensions")
            if len(grid.cells) != self.height or any(len(row) != self.width for row in grid.cells):
                raise InvalidGridError("grid cell matrix shape does not match dimensions")
        for clue in self.clues:
            if not self._validate_clue_position(clue):
                raise InvalidClueError(f"Clue {clue.label} is outside the grid")
            if clue.answer and len(clue.answer) != clue.length:
                raise InvalidClueError(
                    f"Clue {clue.label} expects length {clue.length}, got answer {clue.answer!r}"
                )
        return self

    @property
    def current_grid(self) -> Grid:
        return self.grid_history[-1]

    def add_clue(self, clue: Clue) -> None:
        """Add a new clue to the puzzle.

        Crossword numbering commonly reuses a number for across and down clues,
        so uniqueness is checked on the full clue label rather than number alone.
        """
        if any(existing.label == clue.label for existing in self.clues):
            raise InvalidClueError(f"Clue {clue.label} already exists")
        if not self._validate_clue_position(clue):
            raise InvalidClueError(f"Clue {clue.label} position is invalid")
        self.clues.append(clue)

    def _validate_clue_position(self, clue: Clue) -> bool:
        if clue.row >= self.height or clue.col >= self.width:
            return False
        if clue.direction == Direction.ACROSS:
            return clue.col + clue.length <= self.width
        return clue.row + clue.length <= self.height

    def used_cells(self) -> Set[Tuple[int, int]]:
        return {cell for clue in self.clues for cell in clue.cells()}

    def get_current_clue_chars(self, clue: Clue) -> List[Optional[str]]:
        """Return current characters in the grid for a given clue."""
        self._ensure_known_clue(clue)
        return [self.current_grid.cells[row][col].value for row, col in clue.cells()]

    def get_clues_overlapping_with_cell(self, row: int, col: int) -> List[Clue]:
        """Return all clues that occupy ``(row, col)``."""
        return [clue for clue in self.clues if (row, col) in clue.cells()]

    def set_clue_chars(self, clue: Clue, chars: Sequence[str]) -> None:
        """Fill the grid cells for ``clue``.

        Raises ``InvalidGridError`` if the requested letters conflict with
        already-filled crossing letters.
        """
        self._ensure_known_clue(clue)
        if len(chars) != clue.length:
            raise InvalidClueError(f"Expected {clue.length} characters, got {len(chars)}")

        normalized = [self._normalize_char(char) for char in chars]
        new_grid = self.current_grid.model_copy(deep=True)

        for (row, col), char in zip(clue.cells(), normalized):
            existing = new_grid.cells[row][col].value
            if existing is not None and existing != char:
                raise InvalidGridError(
                    f"Conflict at ({row}, {col}): existing {existing!r}, attempted {char!r}"
                )
            new_grid.cells[row][col].value = char

        self.grid_history.append(new_grid)
        clue.answered = True
        self.clue_history.append(self.clues.index(clue))

    def reveal_clue_answer(self, clue: Clue) -> None:
        """Reveal the answer for a specific clue."""
        if not clue.answer:
            raise InvalidClueError("No answer available for this clue")
        self.set_clue_chars(clue, list(clue.answer))

    def reveal_all(self) -> None:
        """Reveal all provided answers. This is only for manual validation."""
        for clue in self.clues:
            if not clue.answered and clue.answer:
                self.reveal_clue_answer(clue)

    def validate_clue_chars(self, clue: Clue) -> bool:
        """Check whether a clue's current grid entry matches its stored answer."""
        if not clue.answer:
            return False
        return self.get_current_clue_chars(clue) == list(clue.answer)

    def validate_all(self) -> bool:
        """Check all clues with stored answers."""
        answer_clues = [clue for clue in self.clues if clue.answer]
        return bool(answer_clues) and all(self.validate_clue_chars(clue) for clue in answer_clues)

    def undo(self) -> None:
        """Undo the last clue fill."""
        if len(self.grid_history) <= 1:
            raise InvalidGridError("No moves to undo")
        self.grid_history.pop()
        if self.clue_history:
            clue_idx = self.clue_history.pop()
            self.clues[clue_idx].answered = False

    def reset(self) -> None:
        """Reset the puzzle to its initial blank grid."""
        self.grid_history = [Grid.empty(width=self.width, height=self.height)]
        self.clue_history = []
        for clue in self.clues:
            clue.answered = False

    def answer_for(self, clue: Clue) -> Optional[str]:
        chars = self.get_current_clue_chars(clue)
        if any(char is None for char in chars):
            return None
        return "".join(char or "" for char in chars)

    def to_matrix(self, blank: str = "#", empty: str = ".") -> List[List[str]]:
        """Return a simple character matrix for solved/partially solved display."""
        used = self.used_cells()
        matrix: List[List[str]] = []
        for r in range(self.height):
            row: List[str] = []
            for c in range(self.width):
                if (r, c) not in used:
                    row.append(blank)
                else:
                    row.append(self.current_grid.cells[r][c].value or empty)
            matrix.append(row)
        return matrix

    def to_lines(self, blank: str = "#", empty: str = ".") -> List[str]:
        return ["".join(row) for row in self.to_matrix(blank=blank, empty=empty)]

    def _ensure_known_clue(self, clue: Clue) -> None:
        if clue not in self.clues:
            raise InvalidClueError("Clue not found in puzzle")

    @staticmethod
    def _normalize_char(value: str) -> str:
        if not isinstance(value, str) or len(value) != 1 or not value.isalpha():
            raise InvalidClueError(f"Invalid crossword character: {value!r}")
        return value.upper()

    def __repr__(self) -> str:
        return f"<CrosswordPuzzle width={self.width} height={self.height} clues={len(self.clues)}>"

    def __str__(self) -> str:
        # Box drawing characters keep the original helper script readable.
        top_left, top_right = "┌", "┐"
        bottom_left, bottom_right = "└", "┘"
        horizontal, vertical = "─", "│"
        unused = "░"

        result = [top_left + (horizontal * 3 * self.width) + top_right]
        used = self.used_cells()
        for row_idx, row in enumerate(self.current_grid.cells):
            formatted = []
            for cell in row:
                if (row_idx, cell.col) not in used:
                    formatted.append(f" {unused} ")
                else:
                    formatted.append(f" {cell.value or ' '} ")
            result.append(vertical + "".join(formatted) + vertical)
        result.append(bottom_left + (horizontal * 3 * self.width) + bottom_right)
        return "\n".join(result)
