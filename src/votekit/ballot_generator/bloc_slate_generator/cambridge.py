"""
Generate ranked preference profiles using the Cambridge model.

The main API functions in this module are:

- `cambridge_profile_generator`: Generates a single preference profile using the Cambridge model.
- `cambridge_profiles_by_bloc_generator`: Generates preference profiles for each bloc using the
    Cambridge model.
"""

import numpy as np
from pathlib import Path
import json
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
    reduced_historical_majority_ballot_pmf: dict[tuple[str, ...], float],
    reduced_historical_minority_ballot_pmf: dict[tuple[str, ...], float],
    majority_bloc: str,
    historical_slate_to_config_slate: dict[
        str, str
    ],  # TODO: in next major release, make sure this naming aligns with how the user sets the
    # majority/minority groups
):
    """
    Sample historical slate ballots for a given bloc using the Cambridge model.

    Args:
        ballots_per_bloc (dict[str, int]): A dictionary mapping bloc names to the number of ballots
            to sample for that bloc.
        bloc (str): The name of the bloc to sample ballots for.
        config (BlocSlateConfig): The configuration object for the bloc-slate ballot generator.
        reduced_historical_majority_ballot_pmf (dict[tuple[str, ...], float]): A dictionary mapping
            ballot types that begin with the historical majority slate to their frequencies
            (i.e. probabilities). Ballots have been reduced to match the number of candidates in
            the config.
        reduced_historical_minority_ballot_pmf (dict[tuple[str, ...], float]): A dictionary mapping
            ballot types that begin with the historical minority slate to their frequencies
            (i.e. probabilities). Ballots have been reduced to match the number of candidates in
            the config.
        majority_bloc (str): The name of the group in the config corresponding to the historical
            majority slate.
        historical_slate_to_config_slate (dict[str, str]): A dictionary mapping slate names in the
            historical data to slate names in the config.

    Returns:
        list[tuple[str, ...]]: A list of slate ballots, where each ballot is a tuple of slate names
            in the order they appear on that ballot.
    """
    n_ballots = ballots_per_bloc[bloc]

    num_ballots_start_with_maj_slate = np.random.binomial(
        n_ballots, p=config.cohesion_df[majority_bloc].loc[bloc]
    )

    num_ballots_start_with_min_slate = n_ballots - num_ballots_start_with_maj_slate

    hist_maj_ballots = list(reduced_historical_majority_ballot_pmf.keys())
    hist_min_ballots = list(reduced_historical_minority_ballot_pmf.keys())

    maj_slate_ballot_indices = np.random.choice(
        len(hist_maj_ballots),
        size=num_ballots_start_with_maj_slate,
        p=list(reduced_historical_majority_ballot_pmf.values()),
    )

    min_slate_ballot_indices = np.random.choice(
        len(hist_min_ballots),
        size=num_ballots_start_with_min_slate,
        p=list(reduced_historical_minority_ballot_pmf.values()),
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


def _reduce_ballot(original_slate_ballot: str, w_count: int, c_count: int):
    """
    Takes a long ballot and reduces it to match the number of candidates in the config.

    Args:
        original_slate_ballot (str): The original ballot to reduce. Slate type.
        w_count (int): The number of candidates in the majority bloc.
        c_count (int): The number of candidates in the minority bloc.

    Returns:
        str: The reduced ballot.
    """
    new_ballot = ""
    for char in original_slate_ballot:
        if char == "W" and w_count > 0:
            new_ballot += "W"
            w_count -= 1
        elif char == "C" and c_count > 0:
            new_ballot += "C"
            c_count -= 1
    return new_ballot


def _reduce_ballot_pmfs(
    historical_majority_ballot_data_path: Path,
    historical_minority_ballot_data_path: Path,
    config: BlocSlateConfig,
    majority_bloc: str,
    minority_bloc: str,
):
    """
    Reduces the ballot PMFs to match the number of candidates in the config.

    Args:
        historical_majority_ballot_data_path (Path): The path to the JSON file containing the historical majority ballot frequencies.
        historical_minority_ballot_data_path (Path): The path to the JSON file containing the historical minority ballot frequencies.
        config (BlocSlateConfig): The configuration object for the bloc-slate ballot generator.
        majority_bloc (str): The name of the group in the config corresponding to the historical
            majority slate.
        minority_bloc (str): The name of the group in the config corresponding to the historical
            minority slate.

    Returns:
        tuple[dict[tuple[str, ...], float], dict[tuple[str, ...], float]]: A tuple containing the
            reduced historical majority ballot PMF and the reduced historical minority ballot PMF.
    """
    w_count = len(config.slate_to_candidates[majority_bloc])
    c_count = len(config.slate_to_candidates[minority_bloc])

    with open(historical_majority_ballot_data_path, "r") as json_file:
        historical_majority_ballot_frequencies = json.load(json_file)
        reduced_historical_majority_ballot_pmf = {}
        for ballot, freq in historical_majority_ballot_frequencies.items():
            reduced_ballot = _reduce_ballot(
                ballot,
                w_count,
                c_count,
            )
            reduced_historical_majority_ballot_pmf[reduced_ballot] = (
                reduced_historical_majority_ballot_pmf.get(reduced_ballot, 0) + freq
            )

    with open(historical_minority_ballot_data_path, "r") as json_file:
        historical_minority_ballot_frequencies = json.load(json_file)
        reduced_historical_minority_ballot_pmf = {}
        for ballot, freq in historical_minority_ballot_frequencies.items():
            reduced_ballot = _reduce_ballot(
                ballot,
                w_count,
                c_count,
            )
            reduced_historical_minority_ballot_pmf[reduced_ballot] = (
                reduced_historical_minority_ballot_pmf.get(reduced_ballot, 0) + freq
            )

    return (
        reduced_historical_majority_ballot_pmf,
        reduced_historical_minority_ballot_pmf,
    )


# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


def _inner_cambridge_sampler(
    config: BlocSlateConfig,
    majority_bloc: str,
    minority_bloc: str,
) -> dict[str, RankProfile]:
    """
    Inner function to generate profiles by bloc using Cambridge model.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        majority_bloc (str): Name of the group in the config corresponding to the historical
            majority group.
        minority_bloc (str): Name of the group in the config corresponding to the historical
            minority group.

    Returns:
        dict[str, RankProfile]: A dictionary whose keys are bloc strings and values are
            ``RankProfile`` objects representing the generated preference profiles for each bloc.
    """
    historical_slate_to_config_slate = {
        "W": majority_bloc,
        "C": minority_bloc,
    }

    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data/"
    historical_majority_ballot_data_path = Path(
        DATA_DIR,
        "Cambridge_09to17_ballot_types_start_with_W_ballots_distribution.json",
    )

    historical_minority_ballot_data_path = Path(
        DATA_DIR,
        "Cambridge_09to17_ballot_types_start_with_C_ballots_distribution.json",
    )

    reduced_historical_majority_ballot_pmf, reduced_historical_minority_ballot_pmf = (
        _reduce_ballot_pmfs(
            historical_majority_ballot_data_path,
            historical_minority_ballot_data_path,
            config,
            majority_bloc,
            minority_bloc,
        )
    )

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
            reduced_historical_majority_ballot_pmf,
            reduced_historical_minority_ballot_pmf,
            majority_bloc,
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
            f"This model currently only supports two slates, but you passed {len(config.slates)}"
        )

    if len(config.blocs) != 2:
        raise ValueError(
            f"This model currently only supports two blocs, but you passed {len(config.blocs)}"
        )

    if set(config.slates) != set(config.blocs):
        raise ValueError(
            f"The slates ({config.slates}) must correspond to the blocs ({config.blocs})."
        )


def _determine_and_validate_majority_and_minority_blocs(
    config: BlocSlateConfig,
    majority_bloc: Optional[str],
    minority_bloc: Optional[str],
) -> tuple[str, str]:
    """
    Determines the majority and minority groups for the Cambridge model.
    Validates that the names are in the config, and that the majority and minority groups are distinct.

    If the majority and minority groups are not provided, they are determined by the bloc proportions.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        majority_bloc (Optional[str]): Name of the group in the config corresponding to the
            historical majority group. Defaults to None, in which case the majority
            group is determined by the bloc proportions.
        minority_bloc (Optional[str]): Name of the group in the config corresponding to the
            historical minority group. Defaults to None, in which case the minority
            group is determined by the bloc proportions.

    Returns:
        tuple[str, str]: A tuple containing the majority and minority groups.

    Raises:
        ValueError: If the groups are not found in the config.
        ValueError: If the groups are the same.
        ValueError: If the bloc proportions are equal and neither group is set.
    """

    if majority_bloc is None and minority_bloc is None:
        blocs = list(config.bloc_proportions.keys())
        proportions = list(config.bloc_proportions.values())

        if proportions[0] > proportions[1]:
            majority_bloc = blocs[0]
            minority_bloc = blocs[1]
        elif proportions[1] > proportions[0]:
            majority_bloc = blocs[1]
            minority_bloc = blocs[0]
        else:
            raise ValueError(
                "The bloc proportions are equal. You must set a majority_bloc and minority_bloc."
            )
    if majority_bloc == minority_bloc:
        raise ValueError(
            f"Majority group {majority_bloc} and minority group {minority_bloc} must be "
            "distinct."
        )
    if minority_bloc is None:
        minority_bloc = [b for b in config.blocs if b != majority_bloc][0]
    elif majority_bloc is None:
        majority_bloc = [b for b in config.blocs if b != minority_bloc][0]

    if majority_bloc not in config.blocs:
        raise ValueError(f"Majority group {majority_bloc} not found in config.blocs.")
    elif minority_bloc not in config.blocs:
        raise ValueError(f"Minority group {minority_bloc} not found in config.blocs.")

    assert majority_bloc is not None and minority_bloc is not None
    return majority_bloc, minority_bloc


# =================================================
# ================= API Functions =================
# =================================================


def cambridge_profiles_by_bloc_generator(
    config: BlocSlateConfig,
    *,
    majority_bloc: Optional[
        str
    ] = None,  # TODO: in next major release, consider using majority_slate instead of majority_bloc
    minority_bloc: Optional[
        str
    ] = None,  # TODO: in next major release, consider using minority_slate instead of minority_bloc
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
        majority_bloc (Optional[str]): Name of the group in the config corresponding to the
            historical majority group. Defaults to None, in which case the majority
            group is determined by the bloc proportions.
        minority_bloc (Optional[str]): Name of the group in the config corresponding to the
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
    majority_bloc, minority_bloc = _determine_and_validate_majority_and_minority_blocs(
        config, majority_bloc, minority_bloc
    )

    pp_by_bloc = _inner_cambridge_sampler(
        config,
        majority_bloc,
        minority_bloc,
    )

    if group_ballots:
        for bloc, profile in pp_by_bloc.items():
            pp_by_bloc[bloc] = profile.group_ballots()
    return pp_by_bloc


def cambridge_profile_generator(
    config: BlocSlateConfig,
    *,
    majority_bloc: Optional[str] = None,
    minority_bloc: Optional[str] = None,
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
        majority_bloc (str): Name of the group in the config corresponding to the historical
            majority group. Defaults to None, in which case the majority
            group is determined by the bloc proportions.
        minority_bloc (str): Name of the group in the config corresponding to the historical
            minority group. Defaults to None, in which case the minority
            group is determined by the bloc proportions.
        group_ballots (bool): If True, groups identical ballots in the resulting profiles.
            Defaults to True.


    Returns:
        RankProfile: A RankProfile object representing the aggregated generated preference profiles.
    """
    config.is_valid(raise_errors=True)
    _validate_slates_and_blocs(config)
    majority_bloc, minority_bloc = _determine_and_validate_majority_and_minority_blocs(
        config, majority_bloc, minority_bloc
    )

    pp_by_bloc = _inner_cambridge_sampler(
        config,
        majority_bloc,
        minority_bloc,
    )

    profile = RankProfile()
    for prof in pp_by_bloc.values():
        profile += prof

    if group_ballots:
        profile = profile.group_ballots()

    return profile


# TODO create a more generic historical generator that CS will be a subclass of
# this generator should allow for custom data, any num of slates, and unlinked blocs and slates
