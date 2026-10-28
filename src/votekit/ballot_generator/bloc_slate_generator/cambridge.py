"""
Generate ranked preference profiles using the Cambridge model.

The main API functions in this module are:

- `cambridge_profile_generator`: Generates a single preference profile using the Cambridge model.
- `cambridge_profiles_by_bloc_generator`: Generates preference profiles for each bloc using the
  Cambridge model.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from votekit.ballot_generator.bloc_slate_generator._profile_construction import (
    _apportion_bloc_rows,
    _build_profile_from_rankings,
)
from votekit.ballot_generator.bloc_slate_generator._sampling import BATCH_SIZE
from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig
from votekit.ballot_generator.bloc_slate_generator.slate_utils import (
    _fill_slate_candidate_rankings,
    _prepare_slate_candidate_arrays,
)
from votekit.pref_profile import RankProfile
from votekit.pref_profile.profile_matrix import END_OF_RANKING_ID
from votekit.types import Candidate

# =================================================
# ================= API Functions =================
# =================================================


def cambridge_profile_generator(
    config: BlocSlateConfig,
    *,
    majority_bloc: str | None = None,
    minority_bloc: str | None = None,
    group_ballots: bool = True,
    rng_seed: int | None = None,
) -> RankProfile:
    """
    Generate a ranked profile from historical Cambridge, MA, RCV ballot patterns.

    Each bloc's cohesion determines the probability that a ballot begins with the majority slate.
    Conditional on that choice, its slate sequence is drawn from the corresponding historical
    distribution. Sequences are reduced to the available slate sizes without extending short
    ballots. Within each slate, candidate order follows Plackett-Luce on the bloc's supports;
    zero-support candidates follow in a uniformly shuffled tail when zero support is enabled.

    The model requires two blocs and two slates with corresponding names. Historical symbols W and
    C map to the designated majority and minority groups. Grouping combines identical candidate
    rankings after both the historical sequence and candidate order have been sampled.

    Args:
        config (BlocSlateConfig): Configuration specifying voter count, bloc proportions, slate
            membership, cohesion, and within-slate candidate supports.
        majority_bloc (str | None): Group corresponding to historical symbol W. Defaults to None;
            if neither group is supplied, the larger bloc is the majority. If only one is
            supplied, the other is inferred.
        minority_bloc (str | None): Group corresponding to historical symbol C. Defaults to None.
        group_ballots (bool): Whether to combine identical rankings and count their weights.
            Defaults to True.
        rng_seed (int | None): NumPy seed for reproducible output with identical inputs. Defaults
            to None. Batched candidate sampling can change seeded output from the dataframe path.

    Returns:
        RankProfile: Generated rankings with the full candidate universe in configuration order.
            Historical truncation is preserved, and total ballot weight equals
            ``config.n_voters``.

    Raises:
        ValueError: The configuration does not have two correspondingly named blocs and slates,
            the designated groups are invalid, or equal bloc proportions leave the groups
            ambiguous.
    """
    config.is_valid(raise_errors=True)
    _validate_slates_and_blocs(config)
    majority_bloc, minority_bloc = _determine_and_validate_majority_and_minority_blocs(
        config, majority_bloc, minority_bloc
    )

    candidates = tuple(config.candidates)
    rankings, _ = _sample_cambridge_rankings(
        config, candidates, majority_bloc, minority_bloc, np.random.default_rng(rng_seed)
    )

    return _build_profile_from_rankings(rankings, candidates, group_ballots=group_ballots)


def cambridge_profiles_by_bloc_generator(
    config: BlocSlateConfig,
    *,
    # TODO: in next major release, consider using majority_slate instead of majority_bloc
    majority_bloc: str | None = None,
    # TODO: in next major release, consider using minority_slate instead of minority_bloc
    minority_bloc: str | None = None,
    group_ballots: bool = True,
    rng_seed: int | None = None,
) -> dict[str, RankProfile]:
    """
    Generate one Cambridge-model profile per bloc from a shared set of sampled rankings.

    The distribution and historical truncation follow ``cambridge_profile_generator``. Both public
    functions sample the same rankings for a given configuration and seed. This function groups
    within each bloc when requested and retains every bloc's full candidate universe.

    Args:
        config (BlocSlateConfig): Configuration specifying voter count, bloc proportions, slate
            membership, cohesion, and within-slate candidate supports.
        majority_bloc (str | None): Group corresponding to historical symbol W. Defaults to None;
            if neither group is supplied, the larger bloc is the majority. If only one is
            supplied, the other is inferred.
        minority_bloc (str | None): Group corresponding to historical symbol C. Defaults to None.
        group_ballots (bool): Whether to combine identical rankings within each bloc and count
            their weights. Defaults to True.
        rng_seed (int | None): NumPy seed for reproducible output with identical inputs. Defaults
            to None. Batched candidate sampling can change seeded output from the dataframe path.

    Returns:
        dict[str, RankProfile]: Profiles in configuration bloc order, with voter counts determined
            by Huntington apportionment. Each profile owns its matrix storage.

    Raises:
        ValueError: The configuration does not have two correspondingly named blocs and slates,
            the designated groups are invalid, or equal bloc proportions leave the groups
            ambiguous.
    """
    config.is_valid(raise_errors=True)
    _validate_slates_and_blocs(config)
    majority_bloc, minority_bloc = _determine_and_validate_majority_and_minority_blocs(
        config, majority_bloc, minority_bloc
    )

    candidates = tuple(config.candidates)
    rankings, bloc_rows = _sample_cambridge_rankings(
        config, candidates, majority_bloc, minority_bloc, np.random.default_rng(rng_seed)
    )

    return {
        bloc: _build_profile_from_rankings(rankings[rows], candidates, group_ballots=group_ballots)
        for bloc, rows in bloc_rows.items()
    }


# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


def _sample_cambridge_rankings(
    config: BlocSlateConfig,
    candidates: tuple[Candidate, ...],
    majority_bloc: str,
    minority_bloc: str,
    rng: np.random.Generator,
) -> tuple[NDArray[np.intp], dict[str, slice]]:
    """
    Sample historical slate sequences and fill candidate rankings in consecutive bloc rows.

    Historical distributions are reduced and encoded once for the configuration. Each bloc then
    samples from those tables using its cohesion and fills the slate positions with independently
    drawn within-slate PL orders. Short historical ballots remain short. The returned array owns
    its storage, and its row slices let either public API construct profiles from the same
    samples.

    Args:
        config (BlocSlateConfig): Validated two-bloc, two-slate configuration.
        candidates (tuple[Candidate, ...]): Candidate identities in output-ID order.
        majority_bloc (str): Configuration group corresponding to historical symbol W.
        minority_bloc (str): Configuration group corresponding to historical symbol C.
        rng (np.random.Generator): Stream shared by historical and candidate sampling in bloc
            order.

    Returns:
        tuple[NDArray[np.intp], dict[str, slice]]: Candidate-ID rankings padded with end
            sentinels, and the contiguous source rows for each bloc in configuration order.
    """
    data_directory = Path(__file__).resolve().parent / "data"
    majority_pmf, minority_pmf = _reduce_ballot_pmfs(
        data_directory / "Cambridge_09to17_ballot_types_start_with_W_ballots_distribution.json",
        data_directory / "Cambridge_09to17_ballot_types_start_with_C_ballots_distribution.json",
        config,
        majority_bloc,
        minority_bloc,
    )

    slate_ids = {slate: index for index, slate in enumerate(config.slates)}
    ballot_types = _encode_historical_ballot_types(
        (*majority_pmf, *minority_pmf),
        {"W": slate_ids[majority_bloc], "C": slate_ids[minority_bloc]},
        len(candidates),
    )
    majority_probabilities = np.array(tuple(majority_pmf.values()), dtype=np.float64)
    minority_probabilities = np.array(tuple(minority_pmf.values()), dtype=np.float64)

    bloc_rows = _apportion_bloc_rows(config)
    candidate_to_id = {candidate: index for index, candidate in enumerate(candidates)}
    rankings = np.empty((config.n_voters, len(candidates)), dtype=np.intp)

    for bloc, rows in bloc_rows.items():
        candidate_arrays = _prepare_slate_candidate_arrays(config, bloc, candidate_to_id)
        slate_batches = _sample_historical_slate_sequences(
            ballot_types=ballot_types,
            majority_probabilities=majority_probabilities,
            minority_probabilities=minority_probabilities,
            n_voters=rows.stop - rows.start,
            majority_probability=cast(float, config.cohesion_df.loc[bloc, majority_bloc]),
            rng=rng,
        )
        _fill_slate_candidate_rankings(candidate_arrays, slate_batches, rankings[rows], rng)

    return rankings, bloc_rows


# ====================================================
# ================= Helper Functions =================
# ====================================================


def _encode_historical_ballot_types(
    ballot_types: Sequence[str],
    historical_slate_ids: dict[str, int],
    n_candidates: int,
) -> NDArray[np.intp]:
    """
    Encode reduced historical ballot types as slate IDs with trailing end sentinels.

    Each distinct type is encoded once, allowing sampled ballots to select rows without decoding
    their symbols again. Historical ballots may stop before exhausting either slate; the remaining
    positions retain the end sentinel instead of acquiring additional preferences.

    Args:
        ballot_types (Sequence[str]): Historical W/C strings reduced to the configured slate
            sizes.
        historical_slate_ids (dict[str, int]): Mapping from W and C to configuration slate IDs.
        n_candidates (int): Width of the returned table, including trailing padding.

    Returns:
        NDArray[np.intp]: Owned table with one row per supplied type, in the same order. Unranked
            positions contain ``END_OF_RANKING_ID``.
    """
    sequences = np.full((len(ballot_types), n_candidates), END_OF_RANKING_ID, dtype=np.intp)
    for row, ballot_type in enumerate(ballot_types):
        sequences[row, : len(ballot_type)] = [
            historical_slate_ids[symbol] for symbol in ballot_type
        ]

    return sequences


def _sample_historical_slate_sequences(
    ballot_types: NDArray[np.intp],
    majority_probabilities: NDArray[np.float64],
    minority_probabilities: NDArray[np.float64],
    n_voters: int,
    majority_probability: float,
    rng: np.random.Generator,
) -> Iterator[NDArray[np.intp]]:
    """
    Sample historical ballot types and yield independent batches of encoded slate sequences.

    A binomial draw determines how many ballots begin with the majority slate. Those ballots
    precede the minority-first ballots, and each group samples its own historical distribution.
    All type indices are drawn before candidate placement begins, preserving the historical
    sampler's ordering of binomial and categorical draws. Only the selected integer indices are
    retained for the whole bloc; sequence rows are copied from the table one batch at a time.

    Args:
        ballot_types (NDArray[np.intp]): Encoded types with majority-first rows followed by
            minority-first rows, in their respective probability-vector orders. Read unchanged.
        majority_probabilities (NDArray[np.float64]): Probabilities for the majority-first types.
        minority_probabilities (NDArray[np.float64]): Probabilities for the minority-first types.
        n_voters (int): Number of sequences to sample.
        majority_probability (float): Probability that a ballot begins with the majority slate.
        rng (np.random.Generator): Stream advanced by the binomial draw, then majority-first and
            minority-first type sampling.

    Yields:
        NDArray[np.intp]: Owned batches of slate IDs with trailing end sentinels. Each batch can
            be consumed before the next is allocated without modifying the prepared table.
    """
    n_majority_first = rng.binomial(n_voters, p=majority_probability)
    sampled_type_ids = np.empty(n_voters, dtype=np.intp)
    sampled_type_ids[:n_majority_first] = rng.choice(
        len(majority_probabilities), size=n_majority_first, p=majority_probabilities
    )
    sampled_type_ids[n_majority_first:] = len(majority_probabilities) + rng.choice(
        len(minority_probabilities), size=n_voters - n_majority_first, p=minority_probabilities
    )

    for start in range(0, n_voters, BATCH_SIZE):
        yield ballot_types[sampled_type_ids[start : start + BATCH_SIZE]]


def _reduce_ballot(original_slate_ballot: str, w_count: int, c_count: int) -> str:
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
        elif char not in "WC":
            raise ValueError(f"Encountered unexpected character in ballot ranking:  '{char}'")
    return new_ballot


def _reduce_ballot_pmfs(
    historical_majority_ballot_data_path: Path,
    historical_minority_ballot_data_path: Path,
    config: BlocSlateConfig,
    majority_bloc: str,
    minority_bloc: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Reduces the ballot PMFs to match the number of candidates in the config. Since the Cambridge
    data includes ballots that can be too long, i.e. have more candidates than the given
    configuration, we need to "reduce" the ballots to match the number of candidates in the
    config.

    If a ballot is too long, we remove any further occurrences of the slate name. Thus if we only
    have 2 W and 2 C candidates,"WWWCCWWCC" would be reduced to "WWCC".


    Args:
        historical_majority_ballot_data_path (Path): Path to the JSON file containing historical
            majority ballot frequencies.
        historical_minority_ballot_data_path (Path): Path to the JSON file containing historical
            minority ballot frequencies.
        config (BlocSlateConfig): The configuration object for the bloc-slate ballot generator.
        majority_bloc (str): The name of the group in the config corresponding to the historical
            majority slate.
        minority_bloc (str): The name of the group in the config corresponding to the historical
            minority slate.

    Returns:
        tuple[dict[str, float], dict[str, float]]: A tuple containing the reduced historical
            majority and minority ballot PMFs.
    """
    w_count = len(config.slate_to_candidates[majority_bloc])
    c_count = len(config.slate_to_candidates[minority_bloc])

    with open(historical_majority_ballot_data_path, "r") as json_file:
        historical_majority_ballot_frequencies = json.load(json_file)
        reduced_historical_majority_ballot_pmf: dict[str, float] = {}
        for ballot, freq in historical_majority_ballot_frequencies.items():
            reduced_ballot = _reduce_ballot(ballot, w_count, c_count)
            reduced_historical_majority_ballot_pmf[reduced_ballot] = (
                reduced_historical_majority_ballot_pmf.get(reduced_ballot, 0) + freq
            )

    with open(historical_minority_ballot_data_path, "r") as json_file:
        historical_minority_ballot_frequencies = json.load(json_file)
        reduced_historical_minority_ballot_pmf: dict[str, float] = {}
        for ballot, freq in historical_minority_ballot_frequencies.items():
            reduced_ballot = _reduce_ballot(ballot, w_count, c_count)
            reduced_historical_minority_ballot_pmf[reduced_ballot] = (
                reduced_historical_minority_ballot_pmf.get(reduced_ballot, 0) + freq
            )

    return (
        reduced_historical_majority_ballot_pmf,
        reduced_historical_minority_ballot_pmf,
    )


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
    majority_bloc: str | None,
    minority_bloc: str | None,
) -> tuple[str, str]:
    """
    Determines the majority and minority groups for the Cambridge model. Validates that the names
    are in the config and that the majority and minority groups are distinct.

    If the majority and minority groups are not provided, they are determined by the bloc
    proportions.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        majority_bloc (str | None): Name of the group in the config corresponding to the
            historical majority group. Defaults to None, in which case the majority group is
            determined by the bloc proportions.
        minority_bloc (str | None): Name of the group in the config corresponding to the
            historical minority group. Defaults to None, in which case the minority group is
            determined by the bloc proportions.

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
            f"Majority group {majority_bloc} and minority group {minority_bloc} must be distinct."
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


# TODO create a more generic historical generator that CS will be a subclass of this generator
# should allow for custom data, any num of slates, and unlinked blocs and slates
