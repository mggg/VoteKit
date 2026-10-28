"""
Allocate bloc rows and construct profiles from sampled candidate-ID rankings.
"""

from __future__ import annotations

import apportionment.methods as apportion
import numpy as np
from numpy.typing import NDArray

from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig
from votekit.pref_profile import RankProfile
from votekit.pref_profile.profile_matrix import ProfileVoterWeightData, RankMatrix, WeightKind
from votekit.types import Candidate


def _apportion_bloc_rows(config: BlocSlateConfig) -> dict[str, slice]:
    """
    Assign consecutive output rows to blocs using Huntington apportionment.

    Configuration order determines both the apportionment input order and the row order. Blocs
    assigned no voters retain an empty slice so per-bloc results can preserve their metadata.

    Args:
        config (BlocSlateConfig): Validated bloc proportions and total voter count.

    Returns:
        dict[str, slice]: One contiguous row slice per bloc in configuration order.

    Raises:
        TypeError: The apportionment library returns neither a list nor an integer.
    """
    blocs = tuple(config.blocs)
    bloc_counts = apportion.compute(
        "huntington", [config.bloc_proportions[bloc] for bloc in blocs], config.n_voters
    )
    if isinstance(bloc_counts, int):
        bloc_counts = [bloc_counts]
    elif not isinstance(bloc_counts, list):
        raise TypeError(f"Unexpected type from apportionment: {type(bloc_counts)}")

    bloc_rows: dict[str, slice] = {}
    row_start = 0
    for bloc, n_voters in zip(blocs, bloc_counts, strict=True):
        bloc_rows[bloc] = slice(row_start, row_start + n_voters)
        row_start += n_voters

    return bloc_rows


def _build_profile_from_rankings(
    rankings: NDArray[np.intp],
    candidates: tuple[Candidate, ...],
    *,
    group_ballots: bool,
) -> RankProfile:
    """
    Construct a RankProfile from sampled candidate IDs with unit ballot weights.

    Args:
        rankings (NDArray[np.intp]): Complete rankings encoded by position in ``candidates``.
        candidates (tuple[Candidate, ...]): Candidate identities in their declared order.
        group_ballots (bool): Whether to combine identical rankings in the returned profile.

    Returns:
        RankProfile: Profile with independent ranking storage and the requested grouping.
    """
    voter_weight_data = ProfileVoterWeightData(
        weights=np.ones(len(rankings), dtype=np.float64),
        voter_sets=None,
        weight_kind=WeightKind.FLOAT,
    )

    matrix = RankMatrix.from_candidate_ids(
        voter_weight_data=voter_weight_data,
        rankings=rankings,
        candidates=candidates,
    )

    profile = RankProfile._from_matrix(matrix, max_ranking_length=len(candidates))
    return profile.group_ballots() if group_ballots else profile
