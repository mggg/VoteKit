from typing import TypeAlias

Candidate: TypeAlias = str | int
CandidateFloatDictLike: TypeAlias = dict[Candidate, float] | dict[str, float] | dict[int, float]
