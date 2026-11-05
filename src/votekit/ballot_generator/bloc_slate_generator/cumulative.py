"""
Generate scored preference profiles using the name-Cumulative model.

The main API functions in this module are:

- `name_cumulative_profile_generator`: Generates a single preference profile using the
  name-Cumulative model.
- `name_cumulative_ballot_generator_by_bloc`: Generates preference profiles by bloc using the
  name-Cumulative model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from votekit.ballot_generator.bloc_slate_generator._profile_construction import _apportion_bloc_rows
from votekit.ballot_generator.bloc_slate_generator._sampling import BATCH_SIZE
from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig
from votekit.pref_profile import ScoreProfile
from votekit.pref_profile.pref_profile._candidate_names import _normalize_unique_candidates
from votekit.pref_profile.profile_matrix import ProfileVoterWeightData, ScoreMatrix, WeightKind
from votekit.types import Candidate

# =================================================
# ================= API Functions =================
# =================================================


def name_cumulative_profile_generator(
    config: BlocSlateConfig,
    *,
    total_points: int | None = None,
    group_ballots: bool = True,
    rng_seed: int | None = None,
) -> ScoreProfile:
    """
    Generate a scored profile using the name-Cumulative model.

    Each voter distributes ``total_points`` independent choices among candidates according to
    their bloc's combined preference interval. A candidate's score is the number of times they are
    chosen, so scores follow a multinomial distribution and sum to ``total_points`` per voter.
    Candidates receiving no points are unscored, matching ScoreBallot construction. Every profile
    retains the full configured candidate universe in configuration order, with surrounding
    whitespace stripped from string names.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        total_points (int | None): The total number of points to distribute among candidates.
            Defaults to None, which uses the number of configured candidates.
        group_ballots (bool): If True, groups identical ballots in the resulting profile. Defaults
            to True.
        rng_seed (int | None): Seed for the random number generator. An integer seed produces the
            same output for identical inputs. Defaults to None for nondeterministic results.

    Returns:
        ScoreProfile: A `ScoreProfile` object representing the generated ballots.
    """
    config.is_valid(raise_errors=True)

    if total_points is None:
        total_points = len(config.candidates)
    if total_points <= 0:
        raise ValueError("total_points must be a positive integer")

    candidates = tuple(config.candidates)
    scores, _ = _sample_cumulative_scores(
        config, candidates, total_points, np.random.default_rng(rng_seed)
    )
    return _build_cumulative_profile(scores, candidates, group_ballots=group_ballots)


def name_cumulative_ballot_generator_by_bloc(
    config: BlocSlateConfig,
    *,
    total_points: int | None = None,
    group_ballots: bool = True,
    rng_seed: int | None = None,
) -> dict[str, ScoreProfile]:
    """
    Generate one scored profile per bloc using the name-Cumulative model.

    Each voter distributes ``total_points`` independent choices among candidates according to
    their bloc's combined preference interval. A candidate's score is the number of times they are
    chosen, so scores follow a multinomial distribution and sum to ``total_points`` per voter.
    Candidates receiving no points are unscored, matching ScoreBallot construction. Every profile
    retains the full configured candidate universe in configuration order, with surrounding
    whitespace stripped from string names.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        total_points (int | None): The total number of points to distribute among candidates.
            Defaults to None, which uses the number of configured candidates.
        group_ballots (bool): If True, groups identical ballots in the resulting profile. Defaults
            to True.
        rng_seed (int | None): Seed for the random number generator. An integer seed produces the
            same output for identical inputs. Defaults to None for nondeterministic results.

    Returns:
        dict[str, ScoreProfile]: A dictionary whose keys are bloc strings and values are
            `ScoreProfile` objects representing the generated ballots for each bloc.
    """
    config.is_valid(raise_errors=True)

    if total_points is None:
        total_points = len(config.candidates)
    if total_points <= 0:
        raise ValueError("'total_points' must be a positive integer")

    candidates = tuple(config.candidates)
    scores, bloc_rows = _sample_cumulative_scores(
        config, candidates, total_points, np.random.default_rng(rng_seed)
    )
    return {
        bloc: _build_cumulative_profile(scores[rows], candidates, group_ballots=group_ballots)
        for bloc, rows in bloc_rows.items()
    }


# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


def _sample_cumulative_scores(
    config: BlocSlateConfig,
    candidates: tuple[Candidate, ...],
    total_points: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], dict[str, slice]]:
    """
    Sample multinomial scores into consecutive bloc rows in configuration candidate order.

    Each count row contains the points assigned to one voter. Sampling in batches bounds the
    temporary integer arrays while the output owns one float64 score array for all voters. Counts
    of zero become NaN to preserve the unscored entries produced by ScoreBallot. Bloc slices let
    both public entry points use the same samples without constructing interim profiles.

    Args:
        config (BlocSlateConfig): Validated bloc proportions and candidate preferences.
        candidates (tuple[Candidate, ...]): Configuration candidate names in output-column order,
            before profile name normalization.
        total_points (int): Positive number of choices to sample for each voter.
        rng (np.random.Generator): Stream advanced in configuration bloc order, then voter order.

    Returns:
        tuple[NDArray[np.float64], dict[str, slice]]: Owned score array and contiguous row slices
            for each bloc. The configuration is neither modified nor retained.
    """
    bloc_rows = _apportion_bloc_rows(config)
    supports_by_bloc = config.get_combined_candidate_support_arrays_by_bloc(candidates)
    scores = np.empty((config.n_voters, len(candidates)), dtype=np.float64)

    for bloc, rows in bloc_rows.items():
        probabilities = supports_by_bloc[bloc]

        for start in range(rows.start, rows.stop, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, rows.stop)
            scores[start:stop] = rng.multinomial(total_points, probabilities, size=stop - start)
            batch_scores = scores[start:stop]
            batch_scores[batch_scores == 0] = np.nan

    return scores, bloc_rows


# ====================================================
# ================= Helper Functions =================
# ====================================================


def _build_cumulative_profile(
    scores: NDArray[np.float64],
    candidates: tuple[Candidate, ...],
    *,
    group_ballots: bool,
) -> ScoreProfile:
    """
    Construct an owned score matrix from unit-weight samples and optionally group its rows.

    Candidate names are normalized once at the profile boundary. ScoreMatrix copies the sampled
    scores, so profiles built from separate bloc slices retain independent, read-only storage.
    Grouping uses the matrix's score-row semantics and first-appearance order.

    Args:
        scores (NDArray[np.float64]): Sampled scores in candidate order, with NaN for unscored
            entries. Read without modification.
        candidates (tuple[Candidate, ...]): Full configuration candidate universe in column order.
        group_ballots (bool): Whether to combine equal score rows before wrapping the matrix.

    Returns:
        ScoreProfile: Profile with owned storage and unit or grouped ballot weights.
    """
    candidates = _normalize_unique_candidates(candidates, "cumulative generator candidates")
    matrix = ScoreMatrix(
        voter_weight_data=ProfileVoterWeightData(
            weights=np.ones(len(scores), dtype=np.float64),
            voter_sets=None,
            weight_kind=WeightKind.FLOAT,
        ),
        scores=scores,
        score_columns=candidates,
        candidates=candidates,
    )
    if group_ballots:
        matrix = matrix.group_ballots()

    return ScoreProfile._from_matrix(matrix)
