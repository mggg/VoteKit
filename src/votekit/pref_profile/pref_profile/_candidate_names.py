from __future__ import annotations

from collections.abc import Sequence

from votekit.exceptions import ProfileError
from votekit.types import Candidate
from votekit.utils import _validate_candidate_names


def _normalize_candidate_name(candidate: Candidate) -> Candidate:
    """
    Normalize a candidate name by stripping whitespace if it's a string.
    """
    return candidate.strip() if isinstance(candidate, str) else candidate


def _normalize_unique_candidates(
    candidates: Sequence[Candidate], context: str = "candidates"
) -> tuple[Candidate, ...]:
    """
    Normalize candidate names while preserving order and rejecting collisions.

    Args:
        candidates (Sequence[Candidate]): Candidate names to normalize.
        context (str, optional): Source name used in validation errors. Defaults to "candidates".

    Returns:
        tuple[Candidate, ...]: Normalized candidate names.

    Raises:
        ProfileError: Two entries have the same identity after normalization.
        TypeError: Candidates is a bare string or a candidate has an unsupported type.
        ValueError: A normalized name contains a prohibited value or character.
    """
    if isinstance(candidates, str):
        raise TypeError("Candidates cannot be a string. Wrap string in list.")

    normalized_candidates = tuple(_normalize_candidate_name(c) for c in candidates)
    _validate_candidate_names(normalized_candidates, context)

    if len(set(normalized_candidates)) != len(normalized_candidates):
        raise ProfileError(
            "All candidates must be unique. "
            "Two or more candidates have the same identity after stripping whitespace."
        )

    return normalized_candidates
