from typing import Iterable, Mapping, Optional, Sequence, TypeAlias

# ---------------------------------------------------------------------------
# Candidate Types
# ---------------------------------------------------------------------------
Candidate: TypeAlias = str | int
CandidateFloatDictLike: TypeAlias = dict[Candidate, float] | dict[str, float] | dict[int, float]
CandidateListLike: TypeAlias = list[Candidate] | list[str] | list[int]

# ---------------------------------------------------------------------------
# Ballot Types: Ranking and Scores
# ---------------------------------------------------------------------------
Ranking: TypeAlias = Optional[tuple[frozenset[Candidate], ...]]
RankingLike: TypeAlias = Optional[Sequence[Candidate | Iterable[Candidate]]]
ScoresLike: TypeAlias = Optional[
    Mapping[Candidate, float | int] | Mapping[str, float | int] | Mapping[int, float | int]
]

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
