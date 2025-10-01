"""
Generate scored preference profiles using the name-Cumulative model.

The main API functions in this module are:

- `name_cumulative_profile_generator`: Generates a single preference profile using the name-Cumulative
    model.
- `name_cumulative_ballot_generator_by_bloc`: Generates preference profiles by bloc using the
    name-Cumulative model.
"""

import numpy as np
import apportionment.methods as apportion
from typing import Optional

from votekit.ballot import ScoreBallot
from votekit.pref_profile import ScoreProfile
from votekit.ballot_generator.bloc_slate_generator.model import BlocSlateConfig

# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


def _inner_name_cumulative(
    config: BlocSlateConfig, total_points: int
) -> dict[str, ScoreProfile]:
    """
    Inner function to generate cumulative profiles by bloc using the name-Cumulative model.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        total_points (int): The total number of points to distribute among candidates.

    Returns:
        dict[str, ScoreProfile]: A dictionary whose keys are bloc strings and values are
            `ScoreProfile` objects representing the generated ballots for each bloc.
    """
    bloc_lst = config.blocs
    n_voters = int(config.n_voters)

    bloc_counts = apportion.compute(
        "huntington", list(config.bloc_proportions.values()), n_voters
    )
    if not isinstance(bloc_counts, list):
        bloc_counts = [int(bloc_counts)]

    ballots_per_bloc = dict(zip(bloc_lst, bloc_counts))
    pp_by_bloc: dict[str, ScoreProfile] = {}

    pref_by_bloc = config.get_combined_preference_intervals_by_bloc()
    rng = np.random.default_rng()

    for bloc in bloc_lst:
        num_ballots = int(ballots_per_bloc.get(bloc, 0))
        if num_ballots <= 0:
            pp_by_bloc[bloc] = ScoreProfile()
            continue

        pref = pref_by_bloc[bloc]
        non_zero_cands = list(pref.non_zero_cands)
        if not non_zero_cands:
            pp_by_bloc[bloc] = ScoreProfile()
            continue

        # config.get_combined_preference_intervals_by_bloc() should ensure normalization
        # for the non-zero candidates
        p = np.array([pref.interval[c] for c in non_zero_cands], dtype=float)
        assert abs(p.sum() - 1.0) < 1e-10, "PreferenceInterval not normalized"

        # Vectorized: one multinomial per ballot -> shape (num_ballots, n_cands)
        # Each row sums to n_voters and the entries are counts for each candidate
        counts = rng.multinomial(n=total_points, pvals=p, size=num_ballots)

        ballots = [
            ScoreBallot(scores=dict(zip(non_zero_cands, row.astype(float))), weight=1)
            for row in counts
        ]
        pp_by_bloc[bloc] = ScoreProfile(ballots=tuple(ballots))

    return pp_by_bloc


# =================================================
# ================= API Functions =================
# =================================================


def name_cumulative_profile_generator(
    config: BlocSlateConfig,
    *,
    total_points: Optional[int] = None,
    group_ballots: bool = True,
) -> ScoreProfile:
    """
    Generates a ScoreProfile using the name-Cumulative.

    This model samples with replacement from a combined preference interval and counts candidates
    with multiplicity.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.

    Kwargs:
        total_points (Optional[int]): The total number of points to distribute among candidates.
            If None, defaults to the number of candidates in the configuration. Defaults to None.
        group_ballots (bool): If True, groups identical ballots in the resulting profile.
            Defaults to True.

    Returns:
        ScoreProfile: A `ScoreProfile` object representing the generated ballots.
    """
    config.is_valid(raise_errors=True)

    if total_points is None:
        total_points = len(config.candidates)
    if total_points <= 0:
        raise ValueError("total_points must be a positive integer")

    pp_by_bloc = _inner_name_cumulative(config, total_points=total_points)

    pp = ScoreProfile()
    for profile in pp_by_bloc.values():
        pp += profile

    if group_ballots:
        pp = pp.group_ballots()

    return pp


def name_cumulative_ballot_generator_by_bloc(
    config: BlocSlateConfig,
    *,
    total_points: Optional[int] = None,
    group_ballots: bool = True,
) -> dict[str, ScoreProfile]:
    """
    Generates a dictionary mapping bloc names to ScoreProfiles using the name-Cumulative model.

    This model samples with replacement from a combined preference interval and counts candidates
    with multiplicity.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.

    Kwargs:
        total_points (Optional[int]): The total number of points to distribute among candidates.
            If None, defaults to the number of candidates in the configuration. Defaults to None.
        group_ballots (bool): If True, groups identical ballots in the resulting profile.
            Defaults to True.

    Returns:
        dict[str, ScoreProfile]: A dictionary whose keys are bloc strings and values are
            `ScoreProfile` objects representing the generated ballots for each bloc.
    """
    config.is_valid(raise_errors=True)

    if total_points is None:
        total_points = len(config.candidates)
    if total_points <= 0:
        raise ValueError("'total_points' must be a positive integer")

    pp_by_bloc = _inner_name_cumulative(config, total_points=total_points)

    if group_ballots:
        for bloc in pp_by_bloc:
            pp_by_bloc[bloc] = pp_by_bloc[bloc].group_ballots()

    return pp_by_bloc
