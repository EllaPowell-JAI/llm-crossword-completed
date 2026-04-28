"""Pydantic data models used by the crossword solver."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Direction(str, Enum):
    """A crossword answer direction."""

    ACROSS = "across"
    DOWN = "down"


class Cell(BaseModel):
    """A single crossword grid cell."""

    row: int = Field(ge=0)
    col: int = Field(ge=0)
    value: Optional[str] = None

    @field_validator("value")
    @classmethod
    def normalize_value(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if len(value) != 1 or not value.isalpha():
            raise ValueError("cell value must be a single alphabetic character")
        return value.upper()


class Clue(BaseModel):
    """A crossword clue and its grid slot."""

    model_config = ConfigDict(use_enum_values=False)

    number: int = Field(gt=0)
    text: str = Field(min_length=1)
    direction: Direction
    length: int = Field(gt=0)
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    answer: Optional[str] = None
    answered: bool = False

    @field_validator("answer")
    @classmethod
    def normalize_answer(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        letters = "".join(ch for ch in value.upper() if ch.isalpha())
        return letters or None

    def cells(self) -> List[Tuple[int, int]]:
        """Return the ordered ``(row, col)`` coordinates occupied by this clue."""
        if self.direction == Direction.ACROSS:
            return [(self.row, self.col + i) for i in range(self.length)]
        return [(self.row + i, self.col) for i in range(self.length)]

    @property
    def label(self) -> str:
        """A stable, human-readable clue id that disambiguates duplicate numbers."""
        return f"{self.number}{self.direction.value[0].upper()}@{self.row},{self.col}"

    def display_label(self) -> str:
        return f"{self.number} {self.direction.value}"


class Grid(BaseModel):
    """A rectangular crossword grid."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    cells: List[List[Cell]] = Field(default_factory=list)

    def initialize_empty(self) -> None:
        """Initialize an empty grid with blank cells."""
        self.cells = [[Cell(row=r, col=c) for c in range(self.width)] for r in range(self.height)]

    @classmethod
    def empty(cls, width: int, height: int) -> "Grid":
        grid = cls(width=width, height=height)
        grid.initialize_empty()
        return grid
