from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from typing import TypeAlias

# ---------------------------------------------------------------------------
# Candidate Types
# ---------------------------------------------------------------------------
Candidate: TypeAlias = str | int
CandidateFloatDict: TypeAlias = dict[Candidate, float] | dict[str, float] | dict[int, float]
Numeric: TypeAlias = float | Fraction
CandidateNumericDict: TypeAlias = (
    CandidateFloatDict
    | dict[Candidate, int]
    | dict[str, int]
    | dict[int, int]
    | dict[Candidate, Fraction]
    | dict[str, Fraction]
    | dict[int, Fraction]
)
CandidateList: TypeAlias = list[Candidate] | list[str] | list[int]

# ---------------------------------------------------------------------------
# Ballot Types: Ranking and Scores
# ---------------------------------------------------------------------------
Ranking: TypeAlias = tuple[frozenset[Candidate], ...] | None
RankingLike: TypeAlias = Sequence[Candidate | Iterable[Candidate]] | None
ScoresLike: TypeAlias = (
    Mapping[Candidate, float | int] | Mapping[str, float | int] | Mapping[int, float | int] | None
)

# ---------------------------------------------------------------------------
# Profile Bar Plot Types
# ---------------------------------------------------------------------------
PlotLabel: TypeAlias = str | int
CandidatePlotLabelMapping: TypeAlias = (
    Mapping[Candidate, PlotLabel] | Mapping[str, PlotLabel] | Mapping[int, PlotLabel]
)

# ---------------------------------------------------------------------------
# Bar Plot Types
# ---------------------------------------------------------------------------
CategoryLabel: TypeAlias = str | int
DataMapping: TypeAlias = (
    Mapping[str, Mapping[CategoryLabel, float]]
    | Mapping[str, Mapping[str, float]]
    | Mapping[str, Mapping[int, float]]
)
CategoryLabelList: TypeAlias = list[CategoryLabel] | list[str] | list[int]
CategoryLabelMapping: TypeAlias = (
    Mapping[CategoryLabel, CategoryLabel]
    | Mapping[str, CategoryLabel]
    | Mapping[int, CategoryLabel]
)
