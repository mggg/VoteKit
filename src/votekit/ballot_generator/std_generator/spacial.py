"""
Generate ranked preference profiles using the Spacial models

The main API functions in this module are:

- `onedim_spacial_profile_generator`: Generates a single preference profile using a one-dimensional
    spacial model.
- `spacial_profile_and_positions_generator`: Generates a single preference profile using a
    multi-dimensional spacial model with voters and candidates distributed according to specified
    distributions.
- `clustered_spacial_profile_and_positions_generator`: Generates a single preference profile using a
    clustered multi-dimensional spacial model where voters are clustered around candidates.
"""

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Optional, Tuple, Callable, Dict, Any, Sequence

from votekit.metrics import euclidean_dist
from votekit.pref_profile import RankProfile

# =================================================
# ================= API Functions =================
# =================================================


def onedim_spacial_profile_generator(
    candidates: Sequence[str],
    number_of_ballots: int,
) -> RankProfile:
    """
    Generates a ranked preference profile where voters and candidates
    are positioned on a one-dimensional line according to a normal
    distribution. Voter preferences are determined by their proximity
    to candidates on this line.

    Args:
        number_of_ballots (int): The number of ballots to generate.

    Returns:
        RankProfile: A ranked preference profile object.
    """
    n_candidates = len(candidates)

    candidate_position_dict = {c: np.random.normal(0, 1) for c in candidates}
    voter_positions = np.random.normal(0, 1, number_of_ballots)

    ballot_pool = np.full(
        (number_of_ballots, n_candidates), frozenset("~"), dtype=object
    )

    for i, vp in enumerate(voter_positions):
        distance_tuples = [
            (c, abs(v - vp)) for c, v, in candidate_position_dict.items()
        ]
        candidate_ranking = np.array(
            [frozenset({t[0]}) for t in sorted(distance_tuples, key=lambda x: x[1])]
        )
        ballot_pool[i] = candidate_ranking

    df = pd.DataFrame(ballot_pool)
    df.index.name = "Ballot Index"
    df.columns = [f"Ranking_{i + 1}" for i in range(n_candidates)]
    df["Weight"] = 1
    df["Voter Set"] = [frozenset()] * len(df)
    return RankProfile(
        candidates=candidates,
        df=df,
        max_ranking_length=n_candidates,
    )


def spacial_profile_and_positions_generator(
    number_of_ballots: int,
    candidates: list[str],
    voter_dist: Callable[..., np.ndarray] = np.random.uniform,
    voter_dist_kwargs: Optional[Dict[str, Any]] = None,
    candidate_dist: Callable[..., np.ndarray] = np.random.uniform,
    candidate_dist_kwargs: Optional[Dict[str, Any]] = None,
    distance: Callable[[np.ndarray, np.ndarray], float] = euclidean_dist,
) -> Tuple[RankProfile, dict[str, np.ndarray], np.ndarray]:
    """
    Samples a metric position for number_of_ballots voters from
    the voter distribution. Samples a metric position for each candidate
    from the input candidate distribution. With sampled
    positions, this method then creates a ranked RankProfile in which
    voter's preferences are consistent with their distances to the candidates
    in the metric space.

    Args:
        number_of_ballots (int): The number of ballots to generate.
        by_bloc (bool): Dummy variable from parent class.

    Returns:
        Tuple[RankProfile, dict[str, numpy.ndarray], numpy.ndarray]:
            A tuple containing the preference profile object,
            a dictionary with each candidate's position in the metric
            space, and a matrix where each row is a single voter's position
            in the metric space.
    """
    if voter_dist_kwargs is None:
        if voter_dist is np.random.uniform:
            voter_dist_kwargs = {"low": 0.0, "high": 1.0, "size": 2.0}
        else:
            voter_dist_kwargs = {}

    try:
        voter_dist(**voter_dist_kwargs)
    except TypeError:
        raise TypeError("Invalid kwargs for the voter distribution.")

    if candidate_dist_kwargs is None:
        if candidate_dist is np.random.uniform:
            candidate_dist_kwargs = {"low": 0.0, "high": 1.0, "size": 2.0}
        else:
            candidate_dist_kwargs = {}

    try:
        candidate_dist(**candidate_dist_kwargs)
    except TypeError:
        raise TypeError("Invalid kwargs for the candidate distribution.")

    try:
        v = voter_dist(**voter_dist_kwargs)
        c = candidate_dist(**candidate_dist_kwargs)
        distance(v, c)
    except TypeError:
        raise TypeError(
            "Distance function is invalid or incompatible "
            "with voter/candidate distributions."
        )

    candidate_position_dict = {
        c: candidate_dist(**candidate_dist_kwargs) for c in candidates
    }
    voter_positions = np.array(
        [voter_dist(**voter_dist_kwargs) for _ in range(number_of_ballots)]
    )

    ballot_pool = np.full((number_of_ballots, len(candidates)), frozenset("~"))

    for i in range(number_of_ballots):
        distance_tuples = [
            (c, distance(voter_positions[i], c_position))
            for c, c_position, in candidate_position_dict.items()
        ]
        candidate_ranking = np.array(
            [frozenset({t[0]}) for t in sorted(distance_tuples, key=lambda x: x[1])]
        )
        ballot_pool[i] = candidate_ranking

    n_candidates = len(candidates)
    df = pd.DataFrame(ballot_pool)
    df.index.name = "Ballot Index"
    df.columns = [f"Ranking_{i + 1}" for i in range(n_candidates)]
    df["Weight"] = 1
    df["Voter Set"] = [frozenset()] * len(df)
    return (
        RankProfile(
            candidates=candidates,
            df=df,
            max_ranking_length=n_candidates,
        ),
        candidate_position_dict,
        voter_positions,
    )


def clustered_spacial_profile_and_positions_generator(
    number_of_ballots: dict[str, int],
    candidates: list[str],
    voter_dist: Callable[..., np.ndarray] = np.random.normal,
    voter_dist_kwargs: Optional[Dict[str, Any]] = None,
    candidate_dist: Callable[..., np.ndarray] = np.random.uniform,
    candidate_dist_kwargs: Optional[Dict[str, Any]] = None,
    distance: Callable[[np.ndarray, np.ndarray], float] = euclidean_dist,
) -> Tuple[RankProfile, dict[str, np.ndarray], np.ndarray]:
    """
    Samples a metric position for each candidate
    from the input candidate distribution. For each candidate, then sample
    number_of_ballots[candidate] metric positions for voters
    which will be centered around the candidate.
    With sampled positions, this method then creates a ranked RankProfile in which
    voter's preferences are consistent with their distances to the candidates
    in the metric space.

    Args:
        number_of_ballots (dict[str, int]): The number of voters attributed
                    to each candidate {candidate string: # voters}.
        by_bloc (bool): Dummy variable from parent class.

    Returns:
        Tuple[RankProfile, dict[str, numpy.ndarray], numpy.ndarray]:
            A tuple containing the preference profile object,
            a dictionary with each candidate's position in the metric
            space, and a matrix where each row is a single voter's position
            in the metric space.
    """
    if voter_dist_kwargs is None:
        if voter_dist is np.random.normal:
            voter_dist_kwargs = {
                "loc": 0,
                "std": np.array(1.0),
                "size": np.array(2.0),
            }
        else:
            voter_dist_kwargs = {}

    if voter_dist.__name__ not in ["normal", "laplace", "logistic", "gumbel"]:
        raise ValueError("Input voter distribution not supported.")

    try:
        voter_dist_kwargs["loc"] = 0
        voter_dist(**voter_dist_kwargs)
    except TypeError:
        raise TypeError("Invalid kwargs for the voter distribution.")

    if candidate_dist_kwargs is None:
        if candidate_dist is np.random.uniform:
            candidate_dist_kwargs = {"low": 0.0, "high": 1.0, "size": 2.0}
        else:
            candidate_dist_kwargs = {}

    try:
        candidate_dist(**candidate_dist_kwargs)
    except TypeError:
        raise TypeError("Invalid kwargs for the candidate distribution.")

    try:
        v = voter_dist(**voter_dist_kwargs)
        c = candidate_dist(**candidate_dist_kwargs)
        distance(v, c)
    except TypeError:
        raise TypeError(
            "Distance function is invalid or incompatible "
            "with voter/candidate distributions."
        )

    candidate_position_dict: dict[str, NDArray] = {
        c: candidate_dist(**candidate_dist_kwargs) for c in candidates
    }

    n_voters = sum(number_of_ballots.values())
    voter_positions = [np.zeros(2) for _ in range(n_voters)]
    vidx = 0
    for c, c_position in candidate_position_dict.items():  # type: ignore
        for _ in range(number_of_ballots[c]):  # type: ignore
            voter_dist_kwargs["loc"] = c_position
            voter_positions[vidx] = voter_dist(**voter_dist_kwargs)
            vidx += 1

    n_candidates = len(candidates)
    ballot_pool = np.full((n_voters, n_candidates), frozenset("~"), dtype=object)
    for i in range(len(voter_positions)):
        v_position = voter_positions[i]
        distance_tuples = [
            (c, distance(v_position, c_position))
            for c, c_position, in candidate_position_dict.items()
        ]
        candidate_ranking = np.array(
            [frozenset({t[0]}) for t in sorted(distance_tuples, key=lambda x: x[1])]
        )
        ballot_pool[i] = candidate_ranking

    voter_positions_array = np.vstack(voter_positions)

    df = pd.DataFrame(ballot_pool)
    df.index.name = "Ballot Index"
    df.columns = [f"Ranking_{i + 1}" for i in range(n_candidates)]
    df["Weight"] = 1
    df["Voter Set"] = [frozenset()] * len(df)
    return (
        RankProfile(
            candidates=candidates,
            df=df,
            max_ranking_length=n_candidates,
        ),
        candidate_position_dict,
        voter_positions_array,
    )
