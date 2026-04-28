"""Domain-specific exceptions for crossword operations."""


class CrosswordError(Exception):
    """Base exception for crossword errors."""


class InvalidGridError(CrosswordError):
    """Raised when grid operations are invalid."""


class InvalidClueError(CrosswordError):
    """Raised when clue operations are invalid."""


class SolverError(CrosswordError):
    """Raised when a puzzle cannot be solved with the available candidates."""
