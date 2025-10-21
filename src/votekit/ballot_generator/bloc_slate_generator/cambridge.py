"""
Generate ranked preference profiles using the Cambridge model.

The main API functions in this module are:

- `cambridge_profile_generator`: Generates a single preference profile using the Cambridge model.
- `cambridge_profiles_by_bloc_generator`: Generates preference profiles for each bloc using the
    Cambridge model.
"""

import numpy as np
from pathlib import Path
import pickle
from typing import Optional
import apportionment.methods as apportion
from votekit.pref_profile import RankProfile
from votekit.ballot_generator.bloc_slate_generator.slate_utils import (
    _convert_slate_ballots_to_profile,
)
from votekit.ballot_generator.bloc_slate_generator.model import BlocSlateConfig

# ====================================================
# ================= Helper Functions =================
# ====================================================


def _sample_historical_slate_ballots(
    ballots_per_bloc: dict[str, int],
    bloc: str,
    config: BlocSlateConfig,
    historical_majority_ballot_frequencies: dict[tuple[str, ...], float],
    historical_minority_ballot_frequencies: dict[tuple[str, ...], float],
    majority_group: str,
    historical_slate_to_config_slate: dict[str, str],
):
    """
    Sample historical slate ballots for a given bloc using the Cambridge model.

    Args:
        ballots_per_bloc (dict[str, int]): A dictionary mapping bloc names to the number of ballots
            to sample for that bloc.
        bloc (str): The name of the bloc to sample ballots for.
        config (BlocSlateConfig): The configuration object for the bloc-slate ballot generator.
        historical_majority_ballot_frequencies (dict[tuple[str, ...], float]): A dictionary mapping
            ballot types that begin with the historical majority slate to their frequencies
            (i.e. probabilities).
        historical_minority_ballot_frequencies (dict[tuple[str, ...], float]): A dictionary mapping
            ballot types that begin with the historical minority slate to their frequencies
            (i.e. probabilities).
        majority_group (str): The name of the group in the config corresponding to the historical
            majority slate.
        historical_slate_to_config_slate (dict[str, str]): A dictionary mapping slate names in the
            historical data to slate names in the config.

    Returns:
        list[tuple[str, ...]]: A list of slate ballots, where each ballot is a tuple of slate names
            in the order they appear on that ballot.
    """
    n_ballots = ballots_per_bloc[bloc]

    num_ballots_start_with_maj_slate = np.random.binomial(
        n_ballots, p=config.cohesion_df[majority_group].loc[bloc]
    )

    num_ballots_start_with_min_slate = n_ballots - num_ballots_start_with_maj_slate

    hist_maj_ballots = list(historical_majority_ballot_frequencies.keys())
    hist_min_ballots = list(historical_minority_ballot_frequencies.keys())

    maj_slate_ballot_indices = np.random.choice(
        len(hist_maj_ballots),
        size=num_ballots_start_with_maj_slate,
        p=list(historical_majority_ballot_frequencies.values()),
    )

    min_slate_ballot_indices = np.random.choice(
        len(hist_min_ballots),
        size=num_ballots_start_with_min_slate,
        p=list(historical_minority_ballot_frequencies.values()),
    )

    slate_ballots = [
        [
            historical_slate_to_config_slate[historical_slate]
            for historical_slate in hist_maj_ballots[slate_ballot_idx]
        ]
        for slate_ballot_idx in maj_slate_ballot_indices
    ]
    slate_ballots += [
        [
            historical_slate_to_config_slate[historical_slate]
            for historical_slate in hist_min_ballots[slate_ballot_idx]
        ]
        for slate_ballot_idx in min_slate_ballot_indices
    ]

    return slate_ballots


# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


def _inner_cambridge_sampler(
    config: BlocSlateConfig,
    majority_group: str,
    minority_group: str,
) -> dict[str, RankProfile]:
    """
    Inner function to generate profiles by bloc using Cambridge model.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        majority_group (str): Name of the group in the config corresponding to the historical
            majority group.
        minority_group (str): Name of the group in the config corresponding to the historical
            minority group.

    Returns:
        dict[str, RankProfile]: A dictionary whose keys are bloc strings and values are
            ``RankProfile`` objects representing the generated preference profiles for each bloc.
    """
    historical_slate_to_config_slate = {
        "W": majority_group,
        "C": minority_group,
    }

    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data/"
    historical_majority_ballot_data_path = Path(
        DATA_DIR,
        "Cambridge_09to17_ballot_types_start_with_W_ballots_distribution.pkl",
    )

    historical_minority_ballot_data_path = Path(
        DATA_DIR,
        "Cambridge_09to17_ballot_types_start_with_C_ballots_distribution.pkl",
    )
    with open(historical_majority_ballot_data_path, "rb") as pickle_file:
        historical_majority_ballot_frequencies = pickle.load(pickle_file)
    with open(historical_minority_ballot_data_path, "rb") as pickle_file:
        historical_minority_ballot_frequencies = pickle.load(pickle_file)

    bloc_counts = apportion.compute(
        "huntington", list(config.bloc_proportions.values()), config.n_voters
    )
    if not isinstance(bloc_counts, list):
        if not isinstance(bloc_counts, int):
            raise TypeError(
                f"Unexpected type from apportionment got {type(bloc_counts)}"
            )

        bloc_counts = [bloc_counts]

    bloc_lst = config.blocs
    ballots_per_bloc = {bloc: bloc_counts[i] for i, bloc in enumerate(bloc_lst)}
    pref_profile_by_bloc = {b: RankProfile() for b in bloc_lst}

    for bloc in bloc_lst:
        slate_ballots = _sample_historical_slate_ballots(
            ballots_per_bloc,
            bloc,
            config,
            historical_majority_ballot_frequencies,
            historical_minority_ballot_frequencies,
            majority_group,
            historical_slate_to_config_slate,
        )

        pref_profile_by_bloc[bloc] = _convert_slate_ballots_to_profile(
            config, bloc, slate_ballots
        )

    return pref_profile_by_bloc


def _validate_slates_and_blocs(
    config: BlocSlateConfig,
) -> None:
    """
    Validates that the slates and blocs correspond to each other.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.

    Raises:
        ValueError: If the number of slates is not 2.
        ValueError: If the number of blocs is not 2.
        ValueError: If the slates do not correspond to the blocs.
    """
    if len(config.slates) != 2:
        raise ValueError(
            f"This model currently only supports two slates, but you \
                          passed {len(config.slates)}"
        )

    if len(config.blocs) != 2:
        raise ValueError(
            f"This model currently only supports two slates, but you \
                          passed {len(config.slates)}"
        )

    if set(config.slates) != set(config.blocs):
        raise ValueError(
            f"The slates ({config.slates}) must correspond to the blocs ({config.blocs})."
        )


def _determine_and_validate_majority_and_minority_groups(
    config: BlocSlateConfig,
    majority_group: Optional[str],
    minority_group: Optional[str],
) -> tuple[str, str]:
    """
    Determines the majority and minority groups for the Cambridge model.
    Validates that the names are in the config, and that the majority and minority groups are distinct.

    If the majority and minority groups are not provided, they are determined by the bloc proportions.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        majority_group (Optional[str]): Name of the group in the config corresponding to the
            historical majority group. Defaults to None, in which case the majority
            group is determined by the bloc proportions.
        minority_group (Optional[str]): Name of the group in the config corresponding to the
            historical minority group. Defaults to None, in which case the minority
            group is determined by the bloc proportions.

    Returns:
        tuple[str, str]: A tuple containing the majority and minority groups.

    Raises:
        ValueError: If the groups are not found in the config.
        ValueError: If the groups are the same.
        ValueError: If the bloc proportions are equal and neither group is set.
    """

    if majority_group is None and minority_group is None:
        blocs = list(config.bloc_proportions.keys())
        proportions = list(config.bloc_proportions.values())

        if proportions[0] > proportions[1]:
            majority_group = blocs[0]
            minority_group = blocs[1]
        elif proportions[1] > proportions[0]:
            majority_group = blocs[1]
            minority_group = blocs[0]
        else:
            raise ValueError(
                "The bloc proportions are equal. You must set a majority_group and minority_group."
            )
    if majority_group == minority_group:
        raise ValueError(
            f"Majority group {majority_group} and minority group {minority_group} must be "
            "distinct."
        )
    if minority_group is None:
        minority_group = [b for b in config.blocs if b != majority_group][0]
    elif majority_group is None:
        majority_group = [b for b in config.blocs if b != minority_group][0]

    if majority_group not in config.blocs:
        raise ValueError(f"Majority group {majority_group} not found in config.blocs.")
    elif minority_group not in config.blocs:
        raise ValueError(f"Minority group {minority_group} not found in config.blocs.")

    assert majority_group is not None and minority_group is not None
    return majority_group, minority_group


# =================================================
# ================= API Functions =================
# =================================================


def cambridge_profiles_by_bloc_generator(
    config: BlocSlateConfig,
    *,
    majority_group: Optional[str] = None,
    minority_group: Optional[str] = None,
    group_ballots: bool = True,
) -> dict[str, RankProfile]:
    """
    Generates a dictionary mapping bloc names to RankProfiles using historical RCV elections
    occurring in Cambridge, MA. The Cambridge data labels candidates with 'W' and 'C' which
    correspond to the majority and minority slates, respectively. This model only works with
    two blocs and two slates, which must have corresponding names.


    Based on cohesion parameters, decides if a voter casts their top choice within a slate.
    Then uses historical data; given their first choice, choose a
    ballot type from the historical distribution.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.


    Kwargs:
        majority_group (Optional[str]): Name of the group in the config corresponding to the
                    historical majority group. Defaults to None, in which case the majority
                    group is determined by the bloc proportions.
        minority_group (Optional[str]): Name of the group in the config corresponding to the
            historical minority group. Defaults to None, in which case the minority
            group is determined by the bloc proportions.
        group_ballots (bool): If True, groups identical ballots in the resulting profiles.
            Defaults to True.

    Returns:
        dict[str, RankProfile]: A dictionary whose keys are bloc strings and values are
            ``RankProfile`` objects representing the generated preference profiles for each bloc.
    """
    config.is_valid(raise_errors=True)
    _validate_slates_and_blocs(config)
    majority_group, minority_group = (
        _determine_and_validate_majority_and_minority_groups(
            config, majority_group, minority_group
        )
    )

    pp_by_bloc = _inner_cambridge_sampler(
        config,
        majority_group,
        minority_group,
    )

    if group_ballots:
        for bloc, profile in pp_by_bloc.items():
            pp_by_bloc[bloc] = profile.group_ballots()
    return pp_by_bloc


def cambridge_profile_generator(
    config: BlocSlateConfig,
    *,
    majority_group: Optional[str] = None,
    minority_group: Optional[str] = None,
    group_ballots: bool = True,
) -> RankProfile:
    """
    Generates a RankProfile using historical RCV elections
    occurring in Cambridge, MA. The Cambridge data labels candidates with 'W' and 'C' which
    correspond to the majority and minority slates, respectively. This model only works with
    two blocs and two slates, which must have corresponding names.


    Based on cohesion parameters, decides if a voter casts their top choice within a slate.
    Then uses historical data; given their first choice, choose a
    ballot type from the historical distribution.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.

    Kwargs:
        majority_group (str): Name of the group in the config corresponding to the historical
            majority group. Defaults to None, in which case the majority
            group is determined by the bloc proportions.
        minority_group (str): Name of the group in the config corresponding to the historical
            minority group. Defaults to None, in which case the minority
            group is determined by the bloc proportions.
        group_ballots (bool): If True, groups identical ballots in the resulting profiles.
            Defaults to True.


    Returns:
        RankProfile: A RankProfile object representing the aggregated generated preference profiles.
    """
    config.is_valid(raise_errors=True)
    _validate_slates_and_blocs(config)
    majority_group, minority_group = (
        _determine_and_validate_majority_and_minority_groups(
            config, majority_group, minority_group
        )
    )

    pp_by_bloc = _inner_cambridge_sampler(
        config,
        majority_group,
        minority_group,
    )

    profile = RankProfile()
    for prof in pp_by_bloc.values():
        profile += prof

    if group_ballots:
        profile = profile.group_ballots()

    return profile


# TODO create a more generic historical generator that CS will be a subclass of
# this generator should allow for custom data, any num of slates, and unlinked blocs and slates
