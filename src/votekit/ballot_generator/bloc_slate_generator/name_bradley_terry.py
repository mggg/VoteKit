"""
Generate ranked preference profiles using the name-BradleyTerry model.

The main API functions in this module are:

- `name_bt_profile_generator`: Generates a single preference profile using the name-BradleyTerry
    model.
- `name_bt_profiles_by_bloc_generator`: Generates preference profiles by bloc using the
    name-BradleyTerry model.
- `name_bt_profile_generator_using_mcmc`: Generates a single preference profile using MCMC
    sampling from the name-BradleyTerry model.
- `name_bt_profiles_by_bloc_generator_using_mcmc`: Generates preference profiles by bloc using
    MCMC sampling from the name-BradleyTerry model.
"""

from __future__ import annotations

import math

import apportionment.methods as apportion
import numpy as np
from numpy.typing import NDArray

from votekit.ballot_generator.bloc_slate_generator._name_bt_mcmc import (
    name_bt_profile_generator_using_mcmc as name_bt_profile_generator_using_mcmc,
)
from votekit.ballot_generator.bloc_slate_generator._name_bt_mcmc import (
    name_bt_profiles_by_bloc_generator_using_mcmc as name_bt_profiles_by_bloc_generator_using_mcmc,
)
from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig
from votekit.ballot_generator.utils import system_memory
from votekit.pref_profile import RankProfile
from votekit.pref_profile.profile_matrix import ProfileVoterWeightData, RankMatrix, WeightKind
from votekit.types import Candidate

_DP_NAME_BT_BATCH_SIZE = 65_536

# =================================================
# ================= API Functions =================
# =================================================


def name_bt_profile_generator(
    config: BlocSlateConfig,
    *,
    group_ballots: bool = True,
    rng_seed: int | None = None,
) -> RankProfile:
    r"""Generate a preference profile using the name-BradleyTerry model.

    A ranking has probability proportional to the product of its pairwise preferences:
    :math:`P(X>Y>Z) \propto P(X>Y)P(X>Z)P(Y>Z)`, where :math:`P(X>Y)=x/(x+y)`.

    The full distribution is defined over all complete rankings of the declared candidates.

    Let :math:`C` contain :math:`n` candidates with positive supports :math:`s_i`. Let

    .. math::

        \pi=(\pi_1,\ldots,\pi_n)

    denote a complete ranking from first to last.

    The pairwise Bradley–Terry probability is

    .. math::

        P(i\succ j)=\frac{s_i}{s_i+s_j}.

    Multiplying the pairwise factors consistent with ranking :math:`\pi` gives its unnormalized
    ranking weight:

    .. math::

        q(\pi) = \prod_{1\le r<t\le n} \frac{s_{\pi_r}}{s_{\pi_r}+s_{\pi_t}}.

    The probability distribution over all complete rankings is then computed by normalizing the
    weights:

    .. math::

        P(\pi) = \frac{q(\pi)} {\sum_{\sigma\in S_C}q(\sigma)}.

    Where :math:`S_C` is the set of all permutations of the candidates.

    For three candidates, the denominator expands to

    .. math::

        Z = s_A^2s_B+s_A^2s_C +s_B^2s_A+s_B^2s_C +s_C^2s_A+s_C^2s_B,

    so, for example,

    .. math::

        P(A\succ B\succ C)=\frac{s_A^2s_B}{Z}.

    Zero support is rejected unless ``config.allow_zero_support_candidates`` is True. When
    enabled, sample each bloc's positive-support candidates from the distribution above, then
    append a uniformly random permutation of its zero-support candidates independently for
    each ballot. Group identical complete rankings only after appending these tails. A single
    zero-support candidate therefore always appears last. This defines an extension of the
    model when multiple candidates have zero support; each bloc must retain positive total
    support. Both prefix and tail sampling use the seeded generator.

    Args:
        config (BlocSlateConfig): Bloc proportions, candidate supports, and voter count.
        group_ballots (bool, optional): Combine identical rankings and sum their weights. Defaults
            to True.
        rng_seed (int | None, optional): Seed for NumPy's random generator. Defaults to None.
            Negative integers use their absolute value.

    Returns:
        RankProfile: Complete rankings with the declared candidate order and total voter weight.

    Raises:
        ValueError: The configuration is invalid or a bloc's positive-support candidate count
            exceeds the subset-mask or array-index width.
        MemoryError: Estimated memory requirements exceed currently available RAM.
    """
    config.is_valid(raise_errors=True)
    candidates = tuple(config.candidates)
    rng = np.random.default_rng(abs(rng_seed) if rng_seed is not None else None)
    rankings, _ = _sample_name_bt_rankings(config, candidates, rng, group_ballots=group_ballots)

    return _build_profile_from_rankings(rankings, candidates, group_ballots=group_ballots)


def name_bt_profiles_by_bloc_generator(
    config: BlocSlateConfig,
    *,
    group_ballots: bool = True,
    rng_seed: int | None = None,
) -> dict[str, RankProfile]:
    """Generate name-BradleyTerry preference profiles separately for each bloc.

    Use the same DP distribution and random stream as ``name_bt_profile_generator``. Blocs follow
    configuration order and share one seeded NumPy generator. When zero support is enabled in the
    configuration, each ballot ranks its bloc's positive-support candidates first and appends an
    independent, uniformly random permutation of the zero-support candidates. Grouping then
    combines identical complete rankings within each bloc. Each bloc must retain positive total
    support.

    Args:
        config (BlocSlateConfig): Bloc proportions, candidate supports, and voter count.
        group_ballots (bool, optional): Combine identical rankings within each bloc. Defaults to
            True.
        rng_seed (int | None, optional): Seed for NumPy's random generator. Defaults to None.
            Negative integers use their absolute value. Seeded results differ from the earlier
            enumeration sampler.

    Returns:
        dict[str, RankProfile]: Profiles in bloc order, each retaining the declared candidates.

    Raises:
        ValueError: The configuration is invalid or a bloc's positive-support candidate count
            exceeds the subset-mask or array-index width.
        MemoryError: Estimated memory requirements exceed currently available RAM.
    """
    config.is_valid(raise_errors=True)
    candidates = tuple(config.candidates)
    rng = np.random.default_rng(abs(rng_seed) if rng_seed is not None else None)
    rankings, bloc_rows = _sample_name_bt_rankings(
        config, candidates, rng, group_ballots=group_ballots
    )

    return {
        bloc: _build_profile_from_rankings(rankings[rows], candidates, group_ballots=group_ballots)
        for bloc, rows in bloc_rows.items()
    }


# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


def _sample_name_bt_rankings(
    config: BlocSlateConfig,
    candidates: tuple[Candidate, ...],
    rng: np.random.Generator,
    *,
    group_ballots: bool,
) -> tuple[NDArray[np.intp], dict[str, slice]]:
    """Sample complete rankings into one array, retaining each bloc's source rows.

    Args:
        config (BlocSlateConfig): Validated configuration with positive total support per bloc.
        candidates (tuple[Candidate, ...]): Candidate identities in matrix column order.
        rng (np.random.Generator): Generator shared across blocs in configuration order.
        group_ballots (bool): Whether to include grouping in the memory estimate.

    Returns:
        tuple[NDArray[np.intp], dict[str, slice]]: Encoded rankings and each bloc's row slice.

    Raises:
        ValueError: A bloc has nonfinite or negative support, or too many positive-support
            candidates for the subset-mask or array-index width.
        MemoryError: Estimated memory requirements exceed currently available RAM.
    """
    blocs = tuple(config.blocs)
    preference_intervals = config.get_combined_preference_intervals_by_bloc()
    max_positive_candidates = max(
        sum(support > 0 for support in interval.interval.values())
        for interval in preference_intervals.values()
    )
    _check_dp_name_bt_memory(
        len(candidates),
        config.n_voters,
        n_positive_candidates=max_positive_candidates,
        group_ballots=group_ballots,
    )

    bloc_counts = apportion.compute(
        "huntington",
        [config.bloc_proportions[bloc] for bloc in blocs],
        config.n_voters,
    )
    if isinstance(bloc_counts, int):
        bloc_counts = [bloc_counts]

    ranking_matrix = np.empty((config.n_voters, len(candidates)), dtype=int)
    bloc_rows: dict[str, slice] = {}
    row_start = 0

    for bloc, n_bloc_voters in zip(blocs, bloc_counts, strict=True):
        rows = slice(row_start, row_start + n_bloc_voters)
        bloc_rows[bloc] = rows
        row_start += n_bloc_voters
        if n_bloc_voters == 0:
            continue

        supports = np.array(
            [preference_intervals[bloc].interval[candidate] for candidate in candidates],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(supports) & (supports >= 0)):
            raise ValueError(f"Bloc {bloc!r} requires finite, nonnegative candidate supports.")
        _sample_bloc_rankings(candidates, supports, ranking_matrix[rows], rng)

    return ranking_matrix, bloc_rows


def _sample_bloc_rankings(
    candidates: tuple[Candidate, ...],
    supports: NDArray[np.float64],
    rankings: NDArray[np.intp],
    rng: np.random.Generator,
) -> None:
    """Fill a bloc's rankings with BT prefixes and independent uniform zero-support tails.

    Args:
        candidates (tuple[Candidate, ...]): Full candidate universe in matrix-ID order.
        supports (NDArray[np.float64]): Aligned, validated supports with a positive total.
        rankings (NDArray[np.intp]): Writable output for this bloc's complete rankings.
        rng (np.random.Generator): Generator advanced by prefix and tail sampling.
    """
    positive_ids = np.flatnonzero(supports > 0)
    zero_ids = np.flatnonzero(supports == 0)
    positive_candidates = tuple(candidates[index] for index in positive_ids)
    n_positive = len(positive_ids)

    cdf_matrix = _build_cdf_matrix(positive_candidates, supports[positive_ids])
    _sample_rankings_from_cdf(cdf_matrix, rankings[:, :n_positive], rng)
    del cdf_matrix  # Tail sampling does not need to retain the DP table.

    if not len(zero_ids):
        return

    for start in range(0, len(rankings), _DP_NAME_BT_BATCH_SIZE):
        batch = rankings[start : start + _DP_NAME_BT_BATCH_SIZE]
        batch[:, :n_positive] = positive_ids[batch[:, :n_positive]]
        tails = np.broadcast_to(zero_ids, (len(batch), len(zero_ids)))
        rng.permuted(tails, axis=1, out=batch[:, n_positive:])


def _build_profile_from_rankings(
    rankings: NDArray[np.intp],
    candidates: tuple[Candidate, ...],
    *,
    group_ballots: bool,
) -> RankProfile:
    """Construct a RankProfile profile from sampled candidate IDs with unit ballot weights.

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


# ====================================================
# ================= Helper Functions =================
# ====================================================


def _check_dp_name_bt_memory(
    n_candidates: int, n_voters: int, *, n_positive_candidates: int, group_ballots: bool
) -> None:
    """Check the estimated peak allocation against currently available RAM.

    Estimate CDF preparation, batched sampling, and profile construction separately, then compare
    the largest peak with available RAM. Blocs share the output array and prepare their CDFs
    sequentially, so only the largest positive-support table is counted. The full candidate count
    determines output and tail-buffer sizes. Grouping assumes every ranking is distinct. This
    check cannot reserve memory against concurrent allocations by other processes.

    Args:
        n_candidates (int): Number of candidates in each complete ranking.
        n_voters (int): Total rows to generate across all blocs.
        n_positive_candidates (int): Largest positive-support candidate count across blocs.
        group_ballots (bool): Whether to include grouping buffers and the grouped profile.

    Raises:
        ValueError: Subset masks or array dimensions exceed the supported integer widths.
        MemoryError: The estimated peak exceeds available RAM.
    """
    max_candidates = min(32, np.iinfo(np.intp).bits - 2)
    if n_positive_candidates > max_candidates:
        raise ValueError(
            f"DP name-BradleyTerry supports at most {max_candidates} positive-support candidates "
            "per bloc with its "
            "uint32 subset masks and platform array-index width."
        )

    float_bytes = np.dtype(np.float64).itemsize
    index_bytes = np.dtype(int).itemsize
    n_subsets = 1 << n_positive_candidates
    largest_subset_count = math.comb(n_positive_candidates, n_positive_candidates // 2)
    ranking_bytes = n_voters * n_candidates * index_bytes
    cdf_bytes = n_subsets * n_positive_candidates * float_bytes

    # Log totals, uint32 masks, uint8 sizes, and the mask selecting a subset size.
    subset_vector_bytes = n_subsets * (float_bytes + 4 + 1 + 1)
    # Consecutive working matrices can coexist, along with indexed vectors and arithmetic results.
    preparation_work_bytes = largest_subset_count * (
        2 * n_positive_candidates * float_bytes + 8 * index_bytes
    )
    preparation_peak = ranking_bytes + cdf_bytes + subset_vector_bytes + preparation_work_bytes

    buffer_rows = min(_DP_NAME_BT_BATCH_SIZE, n_voters)
    sampling_buffer_bytes = buffer_rows * (
        n_candidates * (float_bytes + 1) + 2 * index_bytes + 2 * float_bytes
    )
    sampling_peak = ranking_bytes + cdf_bytes + n_voters * index_bytes + sampling_buffer_bytes

    # Include owned rankings, candidate-discovery copies, weights, and validation temporaries.
    construction_peak = 4 * ranking_bytes + n_voters * (4 * index_bytes + 4 * float_bytes)
    if group_ballots:
        # Sorting and row selection can retain both profiles plus row keys and index arrays.
        construction_peak = 6 * ranking_bytes + n_voters * (12 * index_bytes + 4 * float_bytes)

    estimated_peak = max(preparation_peak, sampling_peak, construction_peak)
    available_gib = system_memory()["available_gib"]
    if estimated_peak > available_gib * 2**30:
        raise MemoryError(
            f"Not enough memory for DP name-BradleyTerry with {n_candidates} candidates and "
            f"{n_voters} voters. Estimated peak is {estimated_peak / 2**30:.3f} GiB; "
            f"but only {available_gib:.3f} GiB is available. "
            "Reduce the candidate or voter count, or use the MCMC generator."
        )


def _build_cdf_matrix(
    candidates: tuple[Candidate, ...], candidate_supports: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Build conditional next-candidate CDFs for every remaining-candidate subset.

    Args:
        candidates (tuple[Candidate, ...]): Candidate identities in CDF column order.
        candidate_supports (NDArray[np.float64]): Aligned, finite, nonnegative supports with at
            most one zero. The caller must check the preparation memory requirement.

    Returns:
        NDArray[np.float64]: A ``(2**n, n)`` CDF table indexed by subset bitmask. Absent
            candidates have zero probability, and the empty-subset row is unused.
    """
    n_candidates = len(candidates)

    # For a 3-candidate example, the subsets are:
    # [{}, {c1}, {c2}, {c1, c2}, {c3}, {c1, c3}, {c2, c3}, {c1, c2, c3}]
    n_dp_subsets = 2**n_candidates

    # Each candidate gets their own bit in the bitmask
    candidate_bit_assignments = [2**i for i in range(n_candidates)]

    log_supports = np.full(n_candidates, -np.inf, dtype=np.float64)
    np.log(candidate_supports, out=log_supports, where=candidate_supports > 0)

    log_total_suffix_weight = np.full(n_dp_subsets, -np.inf, dtype=np.float64)

    # If there are no candidates remaining to select, then the probability of selecting the empty
    # set is 1.
    log_total_suffix_weight[0] = 0

    cdf_matrix = np.zeros((n_dp_subsets, n_candidates), dtype=np.float64)
    states = np.arange(1 << n_candidates, dtype=np.uint32)
    subset_sizes = np.bitwise_count(states)

    for n_remaining_candidates in range(1, n_candidates + 1):
        states_with_n_remaining_candidates = states[subset_sizes == n_remaining_candidates]
        log_next_cand_weights = np.full(
            (len(states_with_n_remaining_candidates), n_candidates),
            -np.inf,
            dtype=np.float64,
        )

        for cand_id, cand_bit in enumerate(candidate_bit_assignments):
            cand_is_present = (states_with_n_remaining_candidates & cand_bit) != 0

            successor_remaining_cand_set_masks = (
                states_with_n_remaining_candidates[cand_is_present] - cand_bit
            )

            if n_remaining_candidates == 1:
                # The last candidate contributes weight one even when its support is zero.
                log_next_cand_weights[cand_is_present, cand_id] = 0
            else:
                log_next_cand_weights[cand_is_present, cand_id] = (
                    n_remaining_candidates - 1
                ) * log_supports[cand_id] + log_total_suffix_weight[
                    successor_remaining_cand_set_masks
                ]

        log_subset_totals = np.empty(len(states_with_n_remaining_candidates), dtype=np.float64)
        np.logaddexp.reduce(
            log_next_cand_weights,
            axis=1,
            out=log_subset_totals,
        )
        log_total_suffix_weight[states_with_n_remaining_candidates] = log_subset_totals

        # Reuse the working array for probabilities, then cumulative probabilities.
        subset_cdfs = log_next_cand_weights
        subset_cdfs -= log_subset_totals[:, None]

        np.exp(subset_cdfs, out=subset_cdfs)
        np.cumsum(subset_cdfs, axis=1, out=subset_cdfs)

        subset_cdfs /= subset_cdfs[:, -1:]

        cdf_matrix[states_with_n_remaining_candidates] = subset_cdfs

    return cdf_matrix


def _sample_rankings_from_cdf(
    cdf_matrix: NDArray[np.float64],
    rankings: NDArray[np.intp],
    rng: np.random.Generator,
) -> None:
    """Fill encoded rankings using bounded sampling buffers.

    Args:
        cdf_matrix (NDArray[np.float64]): Conditional CDFs indexed by remaining-candidate
            bitmasks, with one column per candidate. The empty-subset row is unused.
        rankings (NDArray[np.intp]): Writable output with one row per voter and one column per
            ranking position. Candidate IDs follow CDF column order.
        rng (np.random.Generator): Generator advanced in position order, then voter order.
    """
    n_voters, n_candidates = rankings.shape
    batch_size = _DP_NAME_BT_BATCH_SIZE
    buffer_rows = min(batch_size, n_voters)
    candidate_bits = np.left_shift(1, np.arange(n_candidates, dtype=int))
    remaining_subset_masks = np.full(n_voters, len(cdf_matrix) - 1, dtype=int)

    voter_cdf_rows = np.empty((buffer_rows, n_candidates), dtype=np.float64)
    passed_thresholds = np.empty((buffer_rows, n_candidates), dtype=bool)
    chosen_candidate_ids = np.empty(buffer_rows, dtype=int)

    # Keep positions outside batches so batching preserves the seeded rankings.
    for position in range(n_candidates):
        for start in range(0, n_voters, batch_size):
            stop = min(start + batch_size, n_voters)
            n_batch_voters = stop - start
            batch_subset_masks = remaining_subset_masks[start:stop]
            np.take(
                cdf_matrix,
                batch_subset_masks,
                axis=0,
                out=voter_cdf_rows[:n_batch_voters],
                mode="clip",
            )

            draws = rng.random(n_batch_voters)
            np.greater_equal(
                draws[:, None],
                voter_cdf_rows[:n_batch_voters],
                out=passed_thresholds[:n_batch_voters],
            )
            np.sum(
                passed_thresholds[:n_batch_voters],
                axis=1,
                dtype=int,
                out=chosen_candidate_ids[:n_batch_voters],
            )

            rankings[start:stop, position] = chosen_candidate_ids[:n_batch_voters]
            batch_subset_masks -= candidate_bits[chosen_candidate_ids[:n_batch_voters]]
