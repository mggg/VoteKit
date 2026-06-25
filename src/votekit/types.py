from typing import TypeAlias

Candidate: TypeAlias = str | int
CandidateFloatDictLike: TypeAlias = dict[Candidate, float] | dict[str, float] | dict[int, float]
CandidateListLike: TypeAlias = list[Candidate] | list[str] | list[int]
