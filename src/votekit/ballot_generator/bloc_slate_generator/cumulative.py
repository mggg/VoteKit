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

from votekit.ballot import ScoreBallot
from votekit.pref_profile import ScoreProfile
from votekit.ballot_generator.bloc_slate_generator.model import BlocSlateConfig

# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


def _inner_name_cumulative(config: BlocSlateConfig) -> dict[str, ScoreProfile]:
    """
    Inner function to generate cumulative profiles by bloc using the name-Cumulative model.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.

    Returns:
        dict[str, ScoreProfile]: A dictionary whose keys are bloc strings and values are
            `ScoreProfile` objects representing the generated ballots for each bloc.
    """
    bloc_lst = config.blocs

    bloc_counts = apportion.compute(
        "huntington", list(config.bloc_proportions.values()), config.n_voters
    )

    if not isinstance(bloc_counts, list):
        if not isinstance(bloc_counts, int):
            raise TypeError(
                f"Unexpected type from apportionment got {type(bloc_counts)}"
            )

        bloc_counts = [bloc_counts]

    ballots_per_bloc = {bloc: bloc_counts[i] for i, bloc in enumerate(bloc_lst)}

    pp_by_bloc = {b: ScoreProfile() for b in bloc_lst}

    pref_interval_by_bloc_dict = config.get_combined_preference_intervals_by_bloc()

    for bloc in bloc_lst:
        ballot_pool = []
        num_ballots = ballots_per_bloc[bloc]
        pref_interval = pref_interval_by_bloc_dict[bloc]

        non_zero_cands = list(pref_interval.non_zero_cands)
        cand_support_vec = [pref_interval.interval[cand] for cand in non_zero_cands]

        for _ in range(num_ballots):
            list_ranking = list(
                np.random.choice(
                    non_zero_cands,
                    config.n_voters,
                    p=cand_support_vec,
                    replace=True,
                )
            )

            scores = {c: 0.0 for c in list_ranking}
            for c in list_ranking:
                scores[c] += 1

            ballot_pool.append(ScoreBallot(scores=scores, weight=1))

        pp = ScoreProfile(ballots=tuple(ballot_pool))
        pp_by_bloc[bloc] = pp

    return pp_by_bloc


# =================================================
# ================= API Functions =================
# =================================================


def name_cumulative_profile_generator(
    config: BlocSlateConfig,
    *,
    group_ballots: bool = True,
) -> ScoreProfile:
    """
    Generates a ScoreProfile using the name-Cumulative.

    This model samples with replacement from a combined preference interval and counts candidates
    with multiplicity.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        group_ballots (bool): If True, groups identical ballots in the resulting profile.
            Defaults to True.

    Returns:
        ScoreProfile: A `ScoreProfile` object representing the generated ballots.
    """
    config.is_valid(raise_errors=True)
    pp_by_bloc = _inner_name_cumulative(config)

    pp = ScoreProfile()
    for profile in pp_by_bloc.values():
        pp += profile

    if group_ballots:
        pp = pp.group_ballots()

    return pp


def name_cumulative_ballot_generator_by_bloc(
    config: BlocSlateConfig,
    *,
    group_ballots: bool = True,
) -> dict[str, ScoreProfile]:
    """
    Generates a dictionary mapping bloc names to ScoreProfiles using the name-Cumulative model.

    This model samples with replacement from a combined preference interval and counts candidates
    with multiplicity.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        group_ballots (bool): If True, groups identical ballots in the resulting profiles.
            Defaults to True.

    Returns:
        dict[str, ScoreProfile]: A dictionary whose keys are bloc strings and values are
            `ScoreProfile` objects representing the generated ballots for each bloc.
    """
    config.is_valid(raise_errors=True)
    pp_by_bloc = _inner_name_cumulative(config)

    if group_ballots:
        for bloc in pp_by_bloc:
            pp_by_bloc[bloc] = pp_by_bloc[bloc].group_ballots()

    return pp_by_bloc
