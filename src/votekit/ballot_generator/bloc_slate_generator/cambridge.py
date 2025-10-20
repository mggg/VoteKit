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
    majority_slate: str,
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
        majority_slate (str): The name of the slate in the config corresponding to the historical
            majority slate.
        historical_slate_to_config_slate (dict[str, str]): A dictionary mapping slate names in the
            historical data to slate names in the config.

    Returns:
        list[tuple[str, ...]]: A list of slate ballots, where each ballot is a tuple of slate names
            in the order they appear on that ballot.
    """
    n_ballots = ballots_per_bloc[bloc]

    num_ballots_start_with_maj_slate = np.random.binomial(
        n_ballots, p=config.cohesion_df[majority_slate].loc[bloc]
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
    historical_majority_ballot_data_path: Path,
    historical_minority_ballot_data_path: Path,
    majority_slate: str,
    minority_slate: str,
    historical_majority_slate: str,
    historical_minority_slate: str,
) -> dict[str, RankProfile]:
    """
    Inner function to generate profiles by bloc using Cambridge model.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        historical_majority_ballot_data_path (Path): File path to an election data file to sample
            from. This should be a pickle file containing a dictionary mapping ballot types
            that begin with the historical majority slate to their frequencies (i.e. probabilities).
        historical_minority_ballot_data_path (Path): File path to an election data file to sample
            from. This should be a pickle file containing a dictionary mapping ballot types
            that begin with the historical minority slate to their frequencies (i.e. probabilities).
        majority_slate (str): Name of the slate in the config corresponding to the historical
            majority slate.
        minority_slate (str): Name of the slate in the config corresponding to the historical
            minority slate.
        historical_majority_slate (str): Name of the slate in the historical data corresponding
            to the majority slate.
        historical_minority_slate (str): Name of the slate in the historical data corresponding
            to the minority slate.

    Returns:
        dict[str, RankProfile]: A dictionary whose keys are bloc strings and values are
            ``RankProfile`` objects representing the generated preference profiles for each bloc.
    """
    historical_slate_to_config_slate = {
        historical_majority_slate: majority_slate,
        historical_minority_slate: minority_slate,
    }

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
            majority_slate,
            historical_slate_to_config_slate,
        )

        pref_profile_by_bloc[bloc] = _convert_slate_ballots_to_profile(
            config, bloc, slate_ballots
        )

    return pref_profile_by_bloc


def _validate_cambridge_slates(
    config: BlocSlateConfig,
    majority_slate: str,
    minority_slate: str,
) -> None:
    """
    Validates the parameters passed to the Cambridge model.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        majority_slate (str): Name of the slate in the config corresponding to the historical
            majority slate.
        minority_slate (str): Name of the slate in the config corresponding to the historical
            minority slate.


    Raises:
        ValueError: If the number of slates is not 2.
        ValueError: If the majority or minority slate is not found in the config.slates.
        ValueError: If majority slate is the same as minority slate.

    """
    if len(config.slates) > 2:
        raise ValueError(
            f"This model currently only supports two slates, but you \
                          passed {len(config.slates)}"
        )

    if majority_slate not in config.slates:
        raise ValueError(f"Majority slate {majority_slate} not found in config.slates")
    if minority_slate not in config.slates:
        raise ValueError(f"Minority slate {minority_slate} not found in config.slates")

    if majority_slate == minority_slate:
        raise ValueError(
            f"Majority slate ({majority_slate}) must be distinct from minority slate "
            f"({minority_slate})."
        )


# =================================================
# ================= API Functions =================
# =================================================


def cambridge_profiles_by_bloc_generator(
    config: BlocSlateConfig,
    majority_slate: str,
    minority_slate: str,
    *,
    historical_majority_ballot_data_path: Optional[Path] = None,
    historical_minority_ballot_data_path: Optional[Path] = None,
    historical_majority_slate: str = "W",
    historical_minority_slate: str = "C",
    group_ballots: bool = True,
) -> dict[str, RankProfile]:
    """
    Generates a dictionary mapping bloc names to RankProfiles using historical RCV elections
    occurring in Cambridge, MA. The Cambridge data labels candidates with 'W' and 'C' which
    correspond to the majority and minority slates, respectively. This model only works with
    two slates.

    Alternative election data can be used if specified. The historical data must be contained
    at the path specified by the 'path' keyword arguments, and the data must be a pickle file
    containing a dictionary mapping ballot types with two slate labels to their frequencies.

    Based on cohesion parameters, decides if a voter casts their top choice within a slate.
    Then uses historical data; given their first choice, choose a
    ballot type from the historical distribution.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        majority_slate (str): Name of the slate in the config corresponding to the historical
            majority slate.
        minority_slate (str): Name of the slate in the config corresponding to the historical
            minority slate.

    Kwargs:
        historical_majority_ballot_data_path (Path, optional): File path to an election data file
            to sample from. This should be a pickle file containing a dictionary mapping ballot
            types that begin with the historical majority slate to their frequencies
            (i.e. probabilities). Defaults to None. If None, will default to Cambridge data that
            ships with VoteKit.
        historical_minority_ballot_data_path (Path, optional): File path to an election data file
            to sample from. This should be a pickle file containing a dictionary mapping ballot
            types that begin with the historical minority slate to their frequencies
            (i.e. probabilities). Defaults to None. If None, will default to Cambridge data that
            ships with VoteKit.
        historical_majority_slate (str): Name of the slate in the historical data
            corresponding to the majority slate. Defaults to "W" for Cambridge.
        historical_minority_slate (str): Name of the slate in the historical data
            corresponding to the minority slate. Defaults to "C" for Cambridge.
        group_ballots (bool): If True, groups identical ballots in the resulting profiles.
            Defaults to True.


    Returns:
        dict[str, RankProfile]: A dictionary whose keys are bloc strings and values are
            ``RankProfile`` objects representing the generated preference profiles for each bloc.
    """
    config.is_valid(raise_errors=True)
    _validate_cambridge_slates(config, majority_slate, minority_slate)

    if historical_majority_ballot_data_path is None:
        BASE_DIR = Path(__file__).resolve().parent
        DATA_DIR = BASE_DIR / "data/"
        historical_majority_ballot_data_path = Path(
            DATA_DIR,
            "Cambridge_09to17_ballot_types_start_with_W_ballots_distribution.pkl",
        )
    if historical_minority_ballot_data_path is None:
        BASE_DIR = Path(__file__).resolve().parent
        DATA_DIR = BASE_DIR / "data/"
        historical_minority_ballot_data_path = Path(
            DATA_DIR,
            "Cambridge_09to17_ballot_types_start_with_C_ballots_distribution.pkl",
        )

    pp_by_bloc = _inner_cambridge_sampler(
        config,
        historical_majority_ballot_data_path=historical_majority_ballot_data_path,
        historical_minority_ballot_data_path=historical_minority_ballot_data_path,
        majority_slate=majority_slate,
        minority_slate=minority_slate,
        historical_majority_slate=historical_majority_slate,
        historical_minority_slate=historical_minority_slate,
    )

    if group_ballots:
        for bloc, profile in pp_by_bloc.items():
            pp_by_bloc[bloc] = profile.group_ballots()
    return pp_by_bloc


def cambridge_profile_generator(
    config: BlocSlateConfig,
    majority_slate: str,
    minority_slate: str,
    *,
    historical_majority_ballot_data_path: Optional[Path] = None,
    historical_minority_ballot_data_path: Optional[Path] = None,
    historical_majority_slate: str = "W",
    historical_minority_slate: str = "C",
    group_ballots: bool = True,
) -> RankProfile:
    """
    Generates a RankProfile using historical RCV elections
    occurring in Cambridge, MA. The Cambridge data labels candidates with 'W' and 'C' which
    correspond to the majority and minority slates, respectively. This model only works with
    two slates.

    Alternative election data can be used if specified. The historical data must be contained
    at the path specified by the 'path' keyword arguments, and the data must be a pickle file
    containing a dictionary mapping ballot types with two slate labels to their frequencies.

    Based on cohesion parameters, decides if a voter casts their top choice within a slate.
    Then uses historical data; given their first choice, choose a
    ballot type from the historical distribution.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        majority_slate (str): Name of the slate in the config corresponding to the historical
            majority slate.
        minority_slate (str): Name of the slate in the config corresponding to the historical
            minority slate.

    Kwargs:
        historical_majority_ballot_data_path (Path, optional): File path to an election data file
            to sample from. This should be a pickle file containing a dictionary mapping ballot
            types that begin with the historical majority slate to their frequencies
            (i.e. probabilities). Defaults to None. If None, will default to Cambridge data that
            ships with VoteKit.
        historical_minority_ballot_data_path (Path, optional): File path to an election data file
            to sample from. This should be a pickle file containing a dictionary mapping ballot
            types that begin with the historical minority slate to their frequencies
            (i.e. probabilities). Defaults to None. If None, will default to Cambridge data that
            ships with VoteKit.
        historical_majority_slate (str): Name of the slate in the historical data
            corresponding to the majority slate. Defaults to "W" for Cambridge.
        historical_minority_slate (str): Name of the slate in the historical data
            corresponding to the minority slate. Defaults to "C" for Cambridge.
        group_ballots (bool): If True, groups identical ballots in the resulting profiles.
            Defaults to True.


    Returns:
        RankProfile: A RankProfile object representing the aggregated generated preference profiles.
    """
    config.is_valid(raise_errors=True)
    _validate_cambridge_slates(config, majority_slate, minority_slate)

    if historical_majority_ballot_data_path is None:
        BASE_DIR = Path(__file__).resolve().parent
        DATA_DIR = BASE_DIR / "data/"
        historical_majority_ballot_data_path = Path(
            DATA_DIR,
            "Cambridge_09to17_ballot_types_start_with_W_ballots_distribution.pkl",
        )
    if historical_minority_ballot_data_path is None:
        BASE_DIR = Path(__file__).resolve().parent
        DATA_DIR = BASE_DIR / "data/"
        historical_minority_ballot_data_path = Path(
            DATA_DIR,
            "Cambridge_09to17_ballot_types_start_with_C_ballots_distribution.pkl",
        )

    pp_by_bloc = _inner_cambridge_sampler(
        config,
        historical_majority_ballot_data_path=historical_majority_ballot_data_path,
        historical_minority_ballot_data_path=historical_minority_ballot_data_path,
        majority_slate=majority_slate,
        minority_slate=minority_slate,
        historical_majority_slate=historical_majority_slate,
        historical_minority_slate=historical_minority_slate,
    )

    profile = RankProfile()
    for prof in pp_by_bloc.values():
        profile += prof

    if group_ballots:
        profile = profile.group_ballots()

    return profile
