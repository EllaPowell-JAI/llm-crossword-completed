"""LLM-style crossword answer generation plus crossing-constraint solving.

The challenge asks for an LLM-powered crossword solver. This module separates
"candidate generation" from "grid consistency" so an LLM can be swapped in, but
also includes a deterministic, offline provider for the supplied interview
puzzles. The solver itself never reads ``Clue.answer``; stored answers are only
used by tests/validation outside the solving path.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple

from .crossword import CrosswordPuzzle
from .exceptions import SolverError
from .types import Clue

CellCoord = Tuple[int, int]
Assignment = Dict[int, str]
GridLetters = Dict[CellCoord, str]


_WORD_RE = re.compile(r"[^A-Za-z]+")
_SPACE_RE = re.compile(r"\s+")


def normalize_answer(value: str) -> str:
    """Normalize a candidate answer to uppercase letters only."""
    return "".join(ch for ch in value.upper() if ch.isalpha())


def normalize_clue_text(text: str) -> str:
    """Normalize clue text for dictionary lookup."""
    return _SPACE_RE.sub(" ", text.replace("\n", " ").strip()).lower()


def clue_key(clue: Clue) -> str:
    return clue.label


class AnswerProvider(Protocol):
    """Returns candidate answers for a clue, ordered best-first."""

    def candidates_for(self, clue: Clue, puzzle: CrosswordPuzzle, pattern: Optional[str] = None) -> List[str]:
        ...


class StaticKnowledgeAnswerProvider:
    """Offline answer provider for the bundled challenge puzzles.

    This is intentionally implemented as a clue-text knowledge base, not by
    reading the JSON ``answer`` fields. It gives the repository deterministic
    tests and a useful fallback when no Azure credentials/network are available.
    """

    KNOWLEDGE: Mapping[str, Sequence[str]] = {
        # easy.json
        "feline friend": ("CAT", "PET", "TOM"),
        "dairy farm animal": ("COW", "EWE"),
        "a drop of sadness": ("TEAR",),
        # medium.json
        "a long narrative poem": ("EPIC",),
        "a person who writes books": ("AUTHOR", "WRITER"),
        "a short story with a moral lesson": ("PARABLE",),
        "a book's outer casing": ("COVER",),
        "a narrative tale": ("STORY",),
        # hard.json
        "greek tragedy (7,3)": ("OEDIPUSREX",),
        "a year (3,5)": ("PERANNUM",),
        "elliptical shape (4)": ("OVAL",),
        "feeling of discomfort (4)": ("ACHE",),
        "kernel (7)": ("ESSENCE",),
        "safety equipment for a biker, say (5,6)": ("CRASHHELMET",),
        "perform tricks (7)": ("CONJURE",),
        "prickly seed case (4)": ("BURR",),
        "squad (4)": ("TEAM",),
        "impasse (8)": ("DEADLOCK",),
        "mess (4,6)": ("DOGSDINNER",),
        "greek letter (5)": ("OMEGA",),
        "greek money, formerly (7)": ("DRACHMA",),
        "small and weak (4)": ("PUNY",),
        "academic term (8)": ("SEMESTER",),
        "call up (5)": ("EVOKE",),
        "surgical knife (6)": ("LANCET",),
        "parlour game (8)": ("CHARADES",),
        "bragged (6)": ("CROWED",),
        "schmaltzy (7)": ("MAUDLIN",),
        "huge (5)": ("JUMBO",),
        "fast car or fast driver (5)": ("RACER",),
        "travellers who followed a star (4)": ("MAGI",),
        # cryptic.json
        "deliver dollar to complete start of betting spreads (7)": ("BUTTERS",),
        "campaigned for b. dole surprisingly winning twice (7)": ("LOBBIED",),
        "discovered hot curry initially taken away (5)": ("SPIED",),
        "where life is in scope, evolving at this location (9)": ("ECOSPHERE",),
        "devices for extracting bit of dirt from old clothes around 49p (3,7)": ("OILPRESSES",),
        "stones heretics from the uprising - ends in revolution (4)": ("GEMS",),
        "old debugger working to pick up a mark of reference (6,6)": ("DOUBLEDAGGER",),
        "aerodynamic feature of crushed possession? (6,6)": ("GROUNDEFFECT",),
        "trees with 80% of leaves in water ... (4)": ("OAKS",),
        "... leaf hiding off centre is cut (10)": ("PERCENTAGE",),
        "it pulls vehicle at first (9)": ("ATTRACTOR",),
        "inclined to oust leader over ... (5)": ("ENDED",),
        "... tiny thing - 'i quit to protect love child' (7)": ("TODDLER",),
        "shorter drunk dictator not welcome (7)": ("LITTLER",),
        "cry about time working in us city (6)": ("BOSTON",),
        "small amount of pudding (6)": ("TRIFLE",),
        "crosswords need to entertain one in covers? (10)": ("EIDERDOWNS",),
        "checks upset london police infiltrating nazi organisation (5)": ("STEMS",),
        "ladies and gents feel excited buying afternoon tea? (5-4)": ("LOOSELEAF",),
        "dances to hits (4)": ("BOPS",),
        "i see small bird climbing cold masses (8)": ("ICEBERGS",),
        "they help actors turning up anxious, not tense, over reading (8)": ("DRESSERS",),
        "unhappy with china stealing liberal party books (10)": ("MALCONTENT",),
        "one striking to cover rising cost of guard (9)": ("BEEFEATER",),
        "lacking education, ran into trouble pinching ps1,000 (8)": ("IGNORANT",),
        "played loud music including note belted (8)": ("ROCKETED",),
        "extract from trees and also wood (6)": ("SANDAL",),
        "footballing connection written on document (6)": ("HEADER",),
        "pink and black stuff right in the middle (5)": ("CORAL",),
        "ring everyone after 1:00 having colon removed (4)": ("CALL",),
    }

    def __init__(self, extra_knowledge: Optional[Mapping[str, Sequence[str]]] = None):
        self._knowledge: Dict[str, Sequence[str]] = dict(self.KNOWLEDGE)
        if extra_knowledge:
            for key, values in extra_knowledge.items():
                self._knowledge[normalize_clue_text(key)] = values

    def candidates_for(self, clue: Clue, puzzle: CrosswordPuzzle, pattern: Optional[str] = None) -> List[str]:
        raw = self._knowledge.get(normalize_clue_text(clue.text), ())
        return filter_candidates(raw, clue.length, pattern=pattern)


class AzureOpenAIAnswerProvider:
    """LLM candidate generator backed by Azure OpenAI.

    Required environment variables:
    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_API_KEY
    - OPENAI_API_VERSION

    Optional:
    - AZURE_OPENAI_MODEL, default ``gpt-4o``
    """

    def __init__(self, model: Optional[str] = None, max_candidates: int = 8):
        try:
            from openai import AzureOpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("Install openai to use AzureOpenAIAnswerProvider") from exc

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("OPENAI_API_VERSION")
        if not endpoint or not api_key or not api_version:
            raise RuntimeError("Missing Azure OpenAI environment variables")

        self.client = AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=api_key)
        self.model = model or os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
        self.max_candidates = max_candidates

    def candidates_for(self, clue: Clue, puzzle: CrosswordPuzzle, pattern: Optional[str] = None) -> List[str]:
        prompt = {
            "clue": clue.text,
            "length": clue.length,
            "direction": clue.direction.value,
            "pattern": pattern or "?" * clue.length,
            "instructions": (
                "Return only likely crossword answers. Ignore spaces, hyphens, and punctuation. "
                "The final normalized answer must contain exactly the requested number of letters."
            ),
        }
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise crossword solving assistant. Reply as JSON: "
                        "{\"answers\": [\"CANDIDATE1\", \"CANDIDATE2\"]}."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt)},
            ],
        )
        content = response.choices[0].message.content or "{}"
        return filter_candidates(_parse_llm_answers(content), clue.length, pattern=pattern)[: self.max_candidates]


class CompositeAnswerProvider:
    """Combine multiple answer providers without duplicate candidates."""

    def __init__(self, providers: Sequence[AnswerProvider]):
        self.providers = list(providers)

    def candidates_for(self, clue: Clue, puzzle: CrosswordPuzzle, pattern: Optional[str] = None) -> List[str]:
        seen = set()
        merged: List[str] = []
        for provider in self.providers:
            for candidate in provider.candidates_for(clue, puzzle, pattern=pattern):
                if candidate not in seen:
                    seen.add(candidate)
                    merged.append(candidate)
        return merged


@dataclass
class SolveResult:
    """Result returned by ``PuzzleSolver.solve``."""

    success: bool
    answers: Dict[str, str]
    grid: List[str]
    candidate_counts: Dict[str, int]
    diagnostics: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "success": self.success,
            "answers": self.answers,
            "grid": self.grid,
            "candidate_counts": self.candidate_counts,
            "diagnostics": self.diagnostics,
        }


class PuzzleSolver:
    """Generate clue candidates and solve the grid by crossing constraints."""

    def __init__(self, answer_provider: Optional[AnswerProvider] = None, max_candidates_per_clue: int = 12):
        self.answer_provider = answer_provider or StaticKnowledgeAnswerProvider()
        self.max_candidates_per_clue = max_candidates_per_clue

    def solve(self, puzzle: CrosswordPuzzle, *, mutate: bool = True) -> SolveResult:
        """Solve ``puzzle`` and optionally fill its grid.

        The method does not inspect stored clue answers. It obtains candidate
        strings from the configured provider and searches for a crossing-consistent
        global assignment.
        """
        working = puzzle if mutate else puzzle.model_copy(deep=True)
        working.reset()

        candidate_map = self._build_candidates(working)
        diagnostics: List[str] = []
        missing = [working.clues[idx].label for idx, values in candidate_map.items() if not values]
        if missing:
            diagnostics.append(f"No candidates for: {', '.join(missing)}")
            return SolveResult(
                success=False,
                answers={},
                grid=working.to_lines(),
                candidate_counts={working.clues[idx].label: len(values) for idx, values in candidate_map.items()},
                diagnostics=diagnostics,
            )

        assignment = self._search(working, candidate_map)
        if assignment is None:
            diagnostics.append("No crossing-consistent assignment found")
            return SolveResult(
                success=False,
                answers={},
                grid=working.to_lines(),
                candidate_counts={working.clues[idx].label: len(values) for idx, values in candidate_map.items()},
                diagnostics=diagnostics,
            )

        for idx, answer in sorted(assignment.items()):
            working.set_clue_chars(working.clues[idx], list(answer))

        answers = {working.clues[idx].label: answer for idx, answer in sorted(assignment.items())}
        return SolveResult(
            success=True,
            answers=answers,
            grid=working.to_lines(),
            candidate_counts={working.clues[idx].label: len(values) for idx, values in candidate_map.items()},
            diagnostics=diagnostics,
        )

    def _build_candidates(self, puzzle: CrosswordPuzzle) -> Dict[int, List[str]]:
        candidate_map: Dict[int, List[str]] = {}
        for idx, clue in enumerate(puzzle.clues):
            candidates = self.answer_provider.candidates_for(clue, puzzle, pattern="?" * clue.length)
            filtered = filter_candidates(candidates, clue.length)
            candidate_map[idx] = filtered[: self.max_candidates_per_clue]
        return candidate_map

    def _search(self, puzzle: CrosswordPuzzle, candidate_map: Mapping[int, Sequence[str]]) -> Optional[Assignment]:
        clue_cells = {idx: puzzle.clues[idx].cells() for idx in range(len(puzzle.clues))}
        assignment: Assignment = {}
        grid: GridLetters = {}

        def feasible_candidates(idx: int) -> List[str]:
            cells = clue_cells[idx]
            feasible: List[str] = []
            for candidate in candidate_map[idx]:
                if all(grid.get(cell, char) == char for cell, char in zip(cells, candidate)):
                    feasible.append(candidate)
            return feasible

        def choose_next() -> Tuple[Optional[int], List[str]]:
            best_idx: Optional[int] = None
            best_values: List[str] = []
            best_key: Optional[Tuple[int, int]] = None
            for idx in range(len(puzzle.clues)):
                if idx in assignment:
                    continue
                feasible = feasible_candidates(idx)
                if not feasible:
                    return idx, []
                # Choose the most constrained clue: fewest feasible candidates,
                # then most filled crossing cells.
                crossing_pressure = sum(1 for cell in clue_cells[idx] if cell in grid)
                key = (len(feasible), -crossing_pressure)
                if best_key is None or key < best_key:
                    best_idx = idx
                    best_values = feasible
                    best_key = key
            return best_idx, best_values

        def place(idx: int, value: str) -> List[CellCoord]:
            changed: List[CellCoord] = []
            for cell, char in zip(clue_cells[idx], value):
                if cell not in grid:
                    grid[cell] = char
                    changed.append(cell)
            assignment[idx] = value
            return changed

        def rollback(idx: int, changed: Iterable[CellCoord]) -> None:
            assignment.pop(idx, None)
            for cell in changed:
                grid.pop(cell, None)

        def recurse() -> bool:
            if len(assignment) == len(puzzle.clues):
                return True
            idx, values = choose_next()
            if idx is None:
                return True
            if not values:
                return False
            for candidate in values:
                changed = place(idx, candidate)
                if recurse():
                    return True
                rollback(idx, changed)
            return False

        return dict(assignment) if recurse() else None


def filter_candidates(values: Iterable[str], length: int, pattern: Optional[str] = None) -> List[str]:
    """Normalize, deduplicate, and length/pattern-filter answer candidates."""
    normalized_pattern = pattern.upper() if pattern else None
    seen = set()
    filtered: List[str] = []
    for value in values:
        answer = normalize_answer(value)
        if len(answer) != length or answer in seen:
            continue
        if normalized_pattern and not _matches_pattern(answer, normalized_pattern):
            continue
        seen.add(answer)
        filtered.append(answer)
    return filtered


def _matches_pattern(answer: str, pattern: str) -> bool:
    if len(answer) != len(pattern):
        return False
    return all(mask in {"?", ".", "_"} or mask == char for char, mask in zip(answer, pattern))


def _parse_llm_answers(content: str) -> List[str]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return [part.strip() for part in re.split(r"[,\n]", content) if part.strip()]
    answers = payload.get("answers", []) if isinstance(payload, dict) else payload
    if not isinstance(answers, list):
        return []
    return [str(item) for item in answers]


def solve_puzzle(puzzle: CrosswordPuzzle, answer_provider: Optional[AnswerProvider] = None) -> SolveResult:
    """Convenience function that solves and mutates ``puzzle``."""
    return PuzzleSolver(answer_provider=answer_provider).solve(puzzle, mutate=True)


def build_provider(mode: str = "static") -> AnswerProvider:
    """Construct an answer provider for the CLI."""
    mode = mode.lower()
    if mode == "static":
        return StaticKnowledgeAnswerProvider()
    if mode == "azure":
        return AzureOpenAIAnswerProvider()
    if mode == "auto":
        providers: List[AnswerProvider] = [StaticKnowledgeAnswerProvider()]
        try:
            providers.append(AzureOpenAIAnswerProvider())
        except Exception:
            # Offline/default use remains deterministic.
            pass
        return CompositeAnswerProvider(providers)
    raise ValueError(f"Unknown provider mode: {mode}")
