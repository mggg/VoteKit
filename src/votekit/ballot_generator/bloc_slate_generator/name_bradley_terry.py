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

import numpy as np
from numpy.typing import NDArray

from votekit.ballot_generator.bloc_slate_generator._name_bt_mcmc import (
    name_bt_profile_generator_using_mcmc as name_bt_profile_generator_using_mcmc,
)
from votekit.ballot_generator.bloc_slate_generator._name_bt_mcmc import (
    name_bt_profiles_by_bloc_generator_using_mcmc as name_bt_profiles_by_bloc_generator_using_mcmc,
)
from votekit.ballot_generator.bloc_slate_generator._profile_construction import (
    _apportion_bloc_rows,
    _build_profile_from_rankings,
)
from votekit.ballot_generator.bloc_slate_generator._sampling import (
    BATCH_SIZE,
    _normalize_log_weights_to_cdf_in_place,
    _sample_cdf_sequences,
)
from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig
from votekit.ballot_generator.utils import system_memory
from votekit.pref_profile import RankProfile
from votekit.types import Candidate

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

    Let :math:`C` contain :math:`m` candidates with positive supports :math:`s_i`. Let

    .. math::

        R=(r_1,\ldots,r_m)\in S_C

    denote a complete ranking from first to last, where :math:`S_C` is the symmetric group on
    :math:`C`, represented here as all orderings of its candidates.

    The pairwise Bradley–Terry probability is

    .. math::

        P(i\succ j)=\frac{s_i}{s_i+s_j}.

    Multiplying the pairwise factors consistent with ranking :math:`R` gives its unnormalized
    ranking weight:

    .. math::

        q(R) = \prod_{1\le a<b\le m} \frac{s_{r_a}}{s_{r_a}+s_{r_b}}.

    The probability distribution over all complete rankings is then computed by normalizing the
    weights:

    .. math::

        P(R) = \frac{q(R)}{\sum_{R'\in S_C}q(R')}.

    Every ranking contains the same pairwise denominator, so that factor cancels when the weights
    are normalized. Equivalently, use the weights

    .. math::

        w(R)=\prod_{j=1}^m s_{r_j}^{m-j},
        \qquad Z(C)=\sum_{R'\in S_C}w(R'),
        \qquad P(R)=\frac{w(R)}{Z(C)}.

    For the three-candidate set :math:`\{A,B,C\}`, this reduced normalizer expands to

    .. math::

        Z(\{A,B,C\}) = s_A^2s_B+s_A^2s_C +s_B^2s_A+s_B^2s_C +s_C^2s_A+s_C^2s_B,

    so, for example,

    .. math::

        P(A\succ B\succ C)=\frac{s_A^2s_B}{Z(\{A,B,C\})}.

    If ``config.allow_zero_support_candidates`` is True, sample the positive-support candidates
    first, then append an independent uniform shuffle of the zero-support candidates to each
    ballot before grouping. This extends the model to multiple zero-support candidates. Each bloc
    must retain positive total support; zero support is otherwise rejected.

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

    Use the same distribution, zero-support policy, and random stream as
    ``name_bt_profile_generator``. Blocs follow configuration order and share one seeded NumPy
    generator. Grouping combines identical complete rankings within each bloc.

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
    """
    Sample complete rankings into one array, retaining each bloc's source rows.

    Huntington apportionment assigns an integer voter count to each bloc before ranking sampling
    begins. Each bloc then uses its own combined preference interval to prepare conditional
    ranking probabilities. Its voters occupy consecutive rows in configuration order, so the
    combined and per-bloc APIs can use the same samples without another draw. All blocs share one
    random generator; a separate DP table is prepared for each interval.

    Args:
        config (BlocSlateConfig): Validated configuration with positive total support per bloc.
        candidates (tuple[Candidate, ...]): Candidate identities in matrix column order.
        rng (np.random.Generator): Generator shared across blocs in configuration order.
        group_ballots (bool): Whether to include grouping in the memory estimate.

    Returns:
        tuple[NDArray[np.intp], dict[str, slice]]: Encoded rankings and each bloc's row slice.

    Raises:
        ValueError: Too many positive-support candidates for the subset-mask or array-index width.
        MemoryError: Estimated memory requirements exceed currently available RAM.
    """
    supports_by_bloc = config.get_combined_candidate_support_arrays_by_bloc(candidates)
    max_positive_candidates = max(
        int(np.count_nonzero(supports > 0)) for supports in supports_by_bloc.values()
    )
    _check_dp_name_bt_memory(
        len(candidates),
        config.n_voters,
        n_positive_candidates=max_positive_candidates,
        group_ballots=group_ballots,
    )

    bloc_rows = _apportion_bloc_rows(config)
    ranking_matrix = np.empty((config.n_voters, len(candidates)), dtype=int)

    for bloc, rows in bloc_rows.items():
        n_bloc_voters = rows.stop - rows.start
        if n_bloc_voters == 0:
            continue

        _sample_bloc_rankings(candidates, supports_by_bloc[bloc], ranking_matrix[rows], rng)

    return ranking_matrix, bloc_rows


def _sample_bloc_rankings(
    candidates: tuple[Candidate, ...],
    supports: NDArray[np.float64],
    rankings: NDArray[np.intp],
    rng: np.random.Generator,
) -> None:
    """
    Sample a bloc's complete rankings from its conditional next-candidate probabilities.

    Prepare the CDF table with ``_build_cdf_matrix`` and draw rankings with
    ``_sample_cdf_sequences``. Map sampled indices back to the full candidate order and apply the
    zero-support tail policy described in ``name_bt_profile_generator`` before returning the
    completed rows for profile construction and optional grouping.

    Args:
        candidates (tuple[Candidate, ...]): Full candidate universe in matrix-ID order.
        supports (NDArray[np.float64]): Aligned, validated supports with a positive total.
        rankings (NDArray[np.intp]): Writable output for this bloc's complete rankings.
        rng (np.random.Generator): Generator advanced by prefix and tail sampling.
    """
    positive_candidate_ids = np.flatnonzero(supports > 0)
    zero_candidate_ids = np.flatnonzero(supports == 0)
    positive_candidates = tuple(candidates[index] for index in positive_candidate_ids)
    n_positive_candidates = len(positive_candidate_ids)

    cdf_matrix = _build_cdf_matrix(positive_candidates, supports[positive_candidate_ids])
    candidate_bits = 1 << np.arange(n_positive_candidates, dtype=np.intp)
    _sample_cdf_sequences(
        cdf_matrix,
        candidate_bits,
        rankings[:, :n_positive_candidates],
        rng,
        batch_size=BATCH_SIZE,
    )
    del cdf_matrix  # Tail sampling does not need to retain the DP table.

    if not len(zero_candidate_ids):
        return

    for start in range(0, len(rankings), BATCH_SIZE):
        batch = rankings[start : start + BATCH_SIZE]
        batch[:, :n_positive_candidates] = positive_candidate_ids[batch[:, :n_positive_candidates]]
        tails = np.broadcast_to(zero_candidate_ids, (len(batch), len(zero_candidate_ids)))
        rng.permuted(tails, axis=1, out=batch[:, n_positive_candidates:])


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
    # Successive selected-mask arrays can coexist; contribution matrices stay batch-sized.
    preparation_rows = min(largest_subset_count, BATCH_SIZE)
    preparation_work_bytes = 2 * largest_subset_count * 4 + preparation_rows * (
        n_positive_candidates * float_bytes + 8 * index_bytes
    )
    preparation_peak = ranking_bytes + cdf_bytes + subset_vector_bytes + preparation_work_bytes

    buffer_rows = min(BATCH_SIZE, n_voters)
    sampling_buffer_bytes = buffer_rows * (
        n_candidates * (float_bytes + 1) + 2 * index_bytes + 2 * float_bytes
    )
    sampling_peak = ranking_bytes + cdf_bytes + n_voters * index_bytes + sampling_buffer_bytes

    # Include owned rankings, candidate-discovery copies, weights, and validation temporaries.
    construction_peak = 4 * ranking_bytes + n_voters * (4 * index_bytes + 4 * float_bytes)
    if group_ballots:
        # Grouping retains raw rows, sorting buffers, and selected rows before matrix
        # construction.
        construction_peak = 5 * ranking_bytes + n_voters * (12 * index_bytes + 4 * float_bytes)

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
    r"""
    Build conditional next-candidate CDFs for every remaining-candidate subset.

    Sampling a ranking one position at a time requires the probability of choosing each candidate
    next, given those still unranked. This table prepares those probabilities for every possible
    remaining subset. Let :math:`C` contain :math:`m` candidates with positive supports
    :math:`s_i`, and let :math:`U\subseteq C` contain the :math:`k` candidates still available.
    Its possible suffix rankings are the elements :math:`R=(r_1,\ldots,r_k)` of the symmetric
    group :math:`S_U`. Because the pairwise BT formula assigns the same denominator to every
    ranking of :math:`U`, that denominator cancels during normalization, leaving the reduced
    ranking weights and their total:

    .. math::

        w(R)=\prod_{j=1}^k s_{r_j}^{k-j},
        \qquad Z(U)=\sum_{R'\in S_U}w(R').

    To compute this total without enumerating every ranking, partition the suffixes by their first
    candidate. A suffix beginning with :math:`i` contributes :math:`s_i^{k-1}` from that position
    because :math:`i` precedes each of the other :math:`k-1` candidates. The remaining factors
    form the weight of a ranking of :math:`U\setminus\{i\}`, so summing over all suffixes
    beginning with :math:`i` gives :math:`s_i^{k-1}Z(U\setminus\{i\})`. Summing these
    contributions over the possible first candidates yields the recurrence

    .. math::

        Z(\varnothing)=1,\qquad
        Z(U)=\sum_{i\in U}s_i^{k-1}Z(U\setminus\{i\}).

    Every term refers to a smaller subset, which determines the order of computation. The empty
    set and each singleton have one ordering with no pairwise comparisons and therefore weight
    one. Starting from these totals, the implementation works through subsets of increasing size,
    reusing the totals already computed. For supports
    :math:`s_A=3`, :math:`s_B=2`, and :math:`s_C=1`, the singleton totals first give

    .. math::

        Z(\{A,B\})=5,\qquad Z(\{A,C\})=4,\qquad Z(\{B,C\})=3.

    Each first choice from the three-candidate set leaves one of these pairs. Their totals are now
    available, so the next calculation combines three stored values to obtain the weight of all
    six complete rankings:

    .. math::

        Z(\{A,B,C\})
        =3^2Z(\{B,C\})+2^2Z(\{A,C\})+1^2Z(\{A,B\})
        =27+16+5=48.

    The same contributions determine the next-candidate probabilities: the weight of all suffixes
    beginning with :math:`i` is divided by the weight of all suffixes. Any fixed prefix
    contributes a common factor that cancels from this ratio, so only the remaining set :math:`U`
    matters:

    .. math::

        p(i\mid U)=\frac{s_i^{k-1}Z(U\setminus\{i\})}{Z(U)}
        \quad (i\in U).

    Although this ratio may be well within float64's range, its numerator and denominator can be
    too small to represent. With 24 candidates each having support :math:`1/24`, for example, a
    complete ranking has weight :math:`(1/24)^{276}`, below the smallest positive float64 value,
    yet each candidate's first-choice probability is :math:`1/24`. Computing the weights directly
    would lose the information needed to recover that probability. The implementation instead
    stores their logarithms, turning products into sums and powers into multiplication. This also
    avoids overflow when unnormalized supports produce very large weights.

    In this representation, ``log_total_suffix_weight`` holds :math:`L(U)=\log Z(U)`, and the
    contribution of candidate :math:`i` becomes

    .. math::

        \ell_i(U)=
        \begin{cases}
            (k-1)\log s_i+L(U\setminus\{i\}), & i\in U,\\
            -\infty, & i\notin U.
        \end{cases}

    These contributions are stored in ``log_next_candidate_weights``. Applying
    ``np.logaddexp.reduce`` computes :math:`L(U)=\log\sum_i\exp(\ell_i(U))` without first
    exponentiating the unnormalized weights. Normalization then takes place by subtraction, giving
    :math:`p(i\mid U)=\exp(\ell_i(U)-L(U))` without forming the potentially unrepresentable
    numerator or denominator. Thus, the earlier example is evaluated using log totals of zero for
    the empty set and singletons, followed by :math:`\log 5`,
    :math:`\log 4`, and :math:`\log 3` for the pairs and :math:`\log 48` for the full set.
    The final probabilities still have float64 precision, so extremely small probabilities may
    round to zero.

    Once these probabilities are known, the sampler needs to locate the row for its current
    remaining subset. Each subset is encoded as an integer bitmask using the candidates'
    zero-based positions in ``candidates``: bit :math:`2^i` is present exactly when candidate
    :math:`i` belongs to :math:`U`. The mask :math:`b(U)=\sum_{i\in U}2^i` therefore assigns
    every subset a unique row number between zero and :math:`2^m-1`.

    For candidate order ``(A, B, C)``, the assigned bits are ``0b001``, ``0b010``, and ``0b100``.
    Since binary notation places the lowest bit on the right, the three bit positions read from
    left to right as C, B, A. The subset :math:`U=\{A,C\}` is therefore ``0b101``, with A and C
    present and B absent:

    .. math::

        b(\{A,C\})=2^0+2^2=1+4=5.

    Its cumulative probabilities occupy ``cdf_matrix[5]``. Choosing A next removes its bit, giving
    ``5 - 1 = 4``, or ``0b100``, the row for :math:`\{C\}`. Subtraction is valid here because the
    chosen candidate's bit is present. Sampling begins with the full set, whose mask is seven
    (``0b111``) in this example, and ends at the empty set, whose mask is zero. Row zero is unused
    because there is no next candidate to select.

    Within each row, probabilities accumulate in candidate order to form the CDF:

    .. math::

        \mathrm{cdf}[b(U),j]=\sum_{i=0}^{j}p(i\mid U).

    Each row retains a column for every candidate, including those absent from :math:`U`. An
    absent candidate contributes no probability, so the cumulative value stays unchanged at its
    column. The final entry should equal one because it includes all possible next choices.
    Floating point accumulation can introduce a small discrepancy, so the implementation divides
    each row by its final entry before storing it.

    The table has :math:`2^m` rows and :math:`m` columns, requiring :math:`8m2^m` bytes of float64
    storage. Preparing its entries takes :math:`O(m2^m)` work and requires temporary buffers in
    addition to the retained table.

    Args:
        candidates (tuple[Candidate, ...]): Candidate identities in CDF column order.
        candidate_supports (NDArray[np.float64]): Aligned, finite, nonnegative supports with at
            most one zero. The caller must check the preparation memory requirement.

    Returns:
        NDArray[np.float64]: A ``(2**m, m)`` CDF table indexed by subset bitmask. Absent
            candidates have zero probability, and the empty-subset row is unused.
    """
    n_candidates = len(candidates)

    # For a 3-candidate example, the subsets are:
    # [{}, {c1}, {c2}, {c1, c2}, {c3}, {c1, c3}, {c2, c3}, {c1, c2, c3}]
    n_dp_subsets = 1 << n_candidates

    log_supports = np.full(n_candidates, -np.inf, dtype=np.float64)
    np.log(candidate_supports, out=log_supports, where=candidate_supports > 0)

    log_total_suffix_weight = np.full(n_dp_subsets, -np.inf, dtype=np.float64)

    # If there are no candidates remaining to select, then the probability of selecting the empty
    # set is 1.
    log_total_suffix_weight[0] = 0

    cdf_matrix = np.zeros((n_dp_subsets, n_candidates), dtype=np.float64)
    states = np.arange(n_dp_subsets, dtype=np.uint32)
    subset_sizes = np.bitwise_count(states)

    for n_remaining_candidates in range(1, n_candidates + 1):
        states_with_n_remaining_candidates = states[subset_sizes == n_remaining_candidates]
        for start in range(0, len(states_with_n_remaining_candidates), BATCH_SIZE):
            subset_masks = states_with_n_remaining_candidates[start : start + BATCH_SIZE]
            _fill_name_bt_cdf_rows(
                subset_masks=subset_masks,
                n_remaining_candidates=n_remaining_candidates,
                log_supports=log_supports,
                log_total_suffix_weight=log_total_suffix_weight,
                cdf_matrix=cdf_matrix,
            )

    return cdf_matrix


def _fill_name_bt_cdf_rows(
    subset_masks: NDArray[np.uint32],
    n_remaining_candidates: int,
    log_supports: NDArray[np.float64],
    log_total_suffix_weight: NDArray[np.float64],
    cdf_matrix: NDArray[np.float64],
) -> None:
    """
    Compute and store suffix totals and next-candidate CDFs for a batch of subsets.

    Choosing a candidate first contributes its support once for every other remaining candidate.
    Multiplying that contribution by the remaining subset's completion weight accounts for all
    suffixes beginning with that choice. In log space, this becomes an addition to the successor
    subset's stored total. All subsets in the batch have the same size, and their successors must
    already have been computed, so the rows can be evaluated together.

    Summing the choice weights gives each subset's completion total. The function stores its
    logarithm for subsequent subsets, then normalizes the choices and accumulates their
    probabilities into a CDF. The temporary choice-weight array is reused throughout this
    conversion and released when the function returns. Only the selected entries in the supplied
    total-weight vector and CDF matrix are modified.

    Args:
        subset_masks (NDArray[np.uint32]): Nonempty subset bitmasks identifying the rows to fill.
            Every mask contains exactly ``n_remaining_candidates`` set bits.
        n_remaining_candidates (int): Number of candidates in each selected subset.
        log_supports (NDArray[np.float64]): Log supports in candidate-ID order. Supports are
            positive except for at most one zero, represented by negative infinity.
        log_total_suffix_weight (NDArray[np.float64]): Writable log completion totals indexed by
            subset bitmask. Successor entries must be initialized, including zero for the empty
            subset. Selected entries are filled in place.
        cdf_matrix (NDArray[np.float64]): Writable table indexed by subset bitmask and candidate
            ID. Selected rows are filled in place, with zero probability for absent candidates and
            a final cumulative probability of one.
    """
    n_candidates = len(log_supports)
    log_next_candidate_weights = np.full(
        (len(subset_masks), n_candidates),
        -np.inf,
        dtype=np.float64,
    )

    candidate_bits = 1 << np.arange(n_candidates, dtype=np.uint32)
    for candidate_id, candidate_bit in enumerate(candidate_bits):
        candidate_is_present = (subset_masks & candidate_bit) != 0
        successor_subset_masks = subset_masks[candidate_is_present] - candidate_bit

        if n_remaining_candidates == 1:
            # The last candidate contributes weight one even when its support is zero.
            log_next_candidate_weights[candidate_is_present, candidate_id] = 0
        else:
            log_first_candidate_weight = (n_remaining_candidates - 1) * log_supports[candidate_id]
            log_next_candidate_weights[candidate_is_present, candidate_id] = (
                log_first_candidate_weight + log_total_suffix_weight[successor_subset_masks]
            )

    log_state_totals = _normalize_log_weights_to_cdf_in_place(log_next_candidate_weights)

    log_total_suffix_weight[subset_masks] = log_state_totals
    cdf_matrix[subset_masks] = log_next_candidate_weights
