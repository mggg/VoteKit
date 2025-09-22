"""
Generate ranked preference profiles using the Cambridge model.

The main API functions in this module are:

- `slate_pl_profile_generator`: Generates a single preference profile using the slate-Plackett-Luce
    model.
- `slate_pl_profiles_by_bloc_generator`: Generates preference profiles by bloc using the
    slate-Plackett-Luce model.
"""

import numpy as np
from pathlib import Path
import pickle
import random
from typing import Optional
import apportionment.methods as apportion

from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile
from votekit.ballot_generator.bloc_slate_generator.model import BlocSlateConfig

# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


def _inner_cambridge_sampler(
    config: BlocSlateConfig,
    path: Path,
    majority_bloc: str,
    minority_bloc: str,
    historical_majority: str,
    historical_minority: str,
) -> dict[str, RankProfile]:
    """
    Args:
        number_of_ballots (int): The number of ballots to generate.
        by_bloc (bool): True if you want the generated profiles returned as a tuple
            ``(pp_by_bloc, pp)``, where ``pp_by_bloc`` is a dictionary with keys = bloc strings
            and values = ``RankProfile`` and ``pp`` is the aggregated profile. False if
            you only want the aggregated profile. Defaults to False.

    Returns:
        Union[RankProfile, Tuple]
    """

    bloc_to_historical = {
        majority_bloc: historical_majority,
        minority_bloc: historical_minority,
    }

    with open(path, "rb") as pickle_file:
        ballot_frequencies = pickle.load(pickle_file)

    cohesion_parameters = {b: config.cohesion_df[b].loc[b] for b in config.blocs}

    # compute the number of bloc and crossover voters in each bloc using Huntington Hill
    voter_types = [
        (b, t) for b in list(config.bloc_proportions.keys()) for t in ["bloc", "cross"]
    ]

    voter_props = [
        (
            cohesion_parameters[b] * config.bloc_proportions[b]
            if t == "bloc"
            else (1 - cohesion_parameters[b]) * config.bloc_proportions[b]
        )
        for b, t in voter_types
    ]

    ballots_per_type = {
        k: int(v)
        for k, v in zip(
            voter_types,
            apportion.compute("huntington", voter_props, config.n_voters),  # type: ignore
        )
    }

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

    pp_by_bloc = {b: RankProfile() for b in bloc_lst}

    # FIX: Change this to use blocs and slates
    for i, bloc in enumerate(bloc_lst):
        bloc_voters = ballots_per_type[(bloc, "bloc")]
        cross_voters = ballots_per_type[(bloc, "cross")]
        ballot_pool = [RankBallot()] * (bloc_voters + cross_voters)

        opp_bloc = bloc_lst[(i + 1) % 2]

        bloc_first_count = sum(
            [
                freq
                for ballot, freq in ballot_frequencies.items()
                if ballot[0] == bloc_to_historical[bloc]
            ]
        )

        opp_bloc_first_count = sum(
            [
                freq
                for ballot, freq in ballot_frequencies.items()
                if ballot[0] == bloc_to_historical[opp_bloc]
            ]
        )

        pref_interval_dict = config.get_combined_preference_intervals_by_bloc()[bloc]

        # compute the relative probabilities of each ballot
        # sorted by ones where the ballot lists the bloc first
        # and those that list the opp first
        prob_ballot_given_bloc_first = {
            ballot: freq / bloc_first_count
            for ballot, freq in ballot_frequencies.items()
            if ballot[0] == bloc_to_historical[bloc]
        }

        prob_ballot_given_opp_first = {
            ballot: freq / opp_bloc_first_count
            for ballot, freq in ballot_frequencies.items()
            if ballot[0] == bloc_to_historical[opp_bloc]
        }

        bloc_voter_ordering = random.choices(
            list(prob_ballot_given_bloc_first.keys()),
            weights=list(prob_ballot_given_bloc_first.values()),
            k=bloc_voters,
        )
        cross_voter_ordering = random.choices(
            list(prob_ballot_given_opp_first.keys()),
            weights=list(prob_ballot_given_opp_first.values()),
            k=cross_voters,
        )

        for i in range(bloc_voters + cross_voters):
            # Based on first choice, randomly choose
            # ballots weighted by Cambridge frequency
            if i < bloc_voters:
                bloc_ordering = bloc_voter_ordering[i]
            else:
                bloc_ordering = cross_voter_ordering[i - bloc_voters]

            pl_ordering = list(
                np.random.choice(
                    list(pref_interval_dict.interval.keys()),
                    len(pref_interval_dict.interval),
                    p=list(pref_interval_dict.interval.values()),
                    replace=False,
                )
            )
            ordered_bloc_slate = [
                c for c in pl_ordering if c in config.slate_to_candidates[bloc]
            ]
            ordered_opp_slate = [
                c for c in pl_ordering if c in config.slate_to_candidates[opp_bloc]
            ]

            # Fill in the bloc slots as determined
            # With the candidate ordering generated with PL
            full_ballot = []
            for b in bloc_ordering:
                if b == bloc_to_historical[bloc]:
                    if ordered_bloc_slate:
                        full_ballot.append(ordered_bloc_slate.pop(0))
                else:
                    if ordered_opp_slate:
                        full_ballot.append(ordered_opp_slate.pop(0))

            ranking = tuple([frozenset({cand}) for cand in full_ballot])
            ballot_pool[i] = RankBallot(ranking=ranking, weight=1)

        pp = RankProfile(ballots=tuple(ballot_pool))
        pp = pp.group_ballots()
        pp_by_bloc[bloc] = pp

    return pp_by_bloc


def _validate_cambridge_blocs(
    config: BlocSlateConfig,
    majority_bloc: Optional[str] = None,
    minority_bloc: Optional[str] = None,
):
    if len(config.slates) > 2:
        raise UserWarning(
            f"This model currently only supports at two blocs, but you \
                          passed {len(config.slates)}"
        )

    if (majority_bloc is None) != (minority_bloc is None):
        raise ValueError(
            "Both W_bloc and C_bloc must be provided or not provided. \
                         You have provided only one."
        )

    elif majority_bloc is not None and majority_bloc == minority_bloc:
        raise ValueError("W and C bloc must be distinct.")

    if majority_bloc is None:
        majority_bloc = [
            bloc for bloc, prop in config.bloc_proportions.items() if prop >= 0.5
        ][0]
    else:
        majority_bloc = majority_bloc

    if minority_bloc is None:
        minority_bloc = [
            bloc for bloc in config.bloc_proportions.keys() if bloc != majority_bloc
        ][0]
    else:
        minority_bloc = minority_bloc

    if set(config.blocs) != set(config.slates):
        raise ValueError(
            "This model requires that a bloc and it's preferred slate have the same name. "
            f"You passed blocs {config.blocs} and slates {config.slates}"
        )

    return majority_bloc, minority_bloc


# =================================================
# ================= API Functions =================
# =================================================


def cambridge_profiles_by_bloc_generator(
    config: BlocSlateConfig,
    *,
    path: Optional[Path] = None,
    majority_bloc: Optional[str] = None,
    minority_bloc: Optional[str] = None,
    # historical_majority: Optional[str] = "W",
    # historical_minority: Optional[str] = "C",
    group_ballots: bool = False,
) -> dict[str, RankProfile]:
    """
    Generates a dictionary mapping bloc names to RankProfiles using historical RCV elections occurring
    in Cambridge, MA.

    Alternative election data can be used if specified. Assumes that there are two
    blocs, a majority and a minorty bloc, which corresponds to the historical Cambridge data.

    Based on cohesion parameters, decides if a voter casts their top choice within their bloc
    or in the opposing bloc. Then uses historical data; given their first choice, to choose a
    ballot type from the historical distribution.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.

    Kwargs:
        path (Optional[Path]): File path to an election data file to sample from. If none, will
            default to Cambridge election data that ships with VoteKit
        majority_bloc (Optional[str]): Name of the bloc corresponding to the majority bloc. Defaults to
            whichever bloc has majority via ``bloc_voter_prop``.
        minority_bloc (Optional[str]): Name of the bloc corresponding to the minority bloc. Defaults to
            whichever bloc has minority via ``bloc_voter_prop``.
        group_ballots (bool): If True, groups identical ballots in the resulting profiles.
            Defaults to False.


    Returns:
        dict[str, RankProfile]: A dictionary whose keys are bloc strings and values are
            ``RankProfile`` objects representing the generated preference profiles for each bloc.
    """
    majority_bloc, minority_bloc = _validate_cambridge_blocs(
        config, majority_bloc=majority_bloc, minority_bloc=minority_bloc
    )

    if path is None:
        BASE_DIR = Path(__file__).resolve().parent
        DATA_DIR = BASE_DIR / "data/"
        path = Path(DATA_DIR, "Cambridge_09to17_ballot_types.p")

    pp_by_bloc = _inner_cambridge_sampler(
        config,
        path=path,
        majority_bloc=majority_bloc,
        minority_bloc=minority_bloc,
        historical_majority="W",
        historical_minority="C",
    )

    if group_ballots:
        for bloc, profile in pp_by_bloc.items():
            pp_by_bloc[bloc] = profile.group_ballots()
    return pp_by_bloc


def cambridge_profile_generator(
    config: BlocSlateConfig,
    *,
    path: Optional[Path] = None,
    majority_bloc: Optional[str] = None,
    minority_bloc: Optional[str] = None,
    # historical_majority: Optional[str] = "W",
    # historical_minority: Optional[str] = "C",
    group_ballots: bool = False,
) -> RankProfile:
    """
    Generates a RankProfile using historical RCV elections occurring in Cambridge, MA.

    Alternative election data can be used if specified. Assumes that there are two
    blocs, a majority and a minorty bloc, which corresponds to the historical Cambridge data.

    Based on cohesion parameters, decides if a voter casts their top choice within their bloc
    or in the opposing bloc. Then uses historical data; given their first choice, to choose a
    ballot type from the historical distribution.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.

    Kwargs:
        path (Optional[Path]): File path to an election data file to sample from. If none, will
            default to Cambridge election data that ships with VoteKit
        majority_bloc (Optional[str]): Name of the bloc corresponding to the majority bloc. Defaults to
            whichever bloc has majority via ``bloc_voter_prop``.
        minority_bloc (Optional[str]): Name of the bloc corresponding to the minority bloc. Defaults to
            whichever bloc has minority via ``bloc_voter_prop``.
        group_ballots (bool): If True, groups identical ballots in the resulting profiles.
            Defaults to False.


    Returns:
        dict[str, RankProfile]: A dictionary whose keys are bloc strings and values are
            ``RankProfile`` objects representing the generated preference profiles for each bloc.
    """
    config.is_valid(raise_errors=True)
    majority_bloc, minority_bloc = _validate_cambridge_blocs(
        config, majority_bloc=majority_bloc, minority_bloc=minority_bloc
    )

    if path is None:
        BASE_DIR = Path(__file__).resolve().parent
        DATA_DIR = BASE_DIR / "data/"
        path = Path(DATA_DIR, "Cambridge_09to17_ballot_types.p")

    pp_by_bloc = _inner_cambridge_sampler(
        config,
        path=path,
        majority_bloc=majority_bloc,
        minority_bloc=minority_bloc,
        historical_majority="W",
        historical_minority="C",
    )

    profile = RankProfile()
    for prof in pp_by_bloc.values():
        profile += prof

    if group_ballots:
        profile = profile.group_ballots()

    return profile
