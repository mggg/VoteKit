"""
Generate ranked preference profiles using the slate-BradleyTerry model.

The main API functions in this module are:

- `slate_bt_profile_generator`: Generates a single preference profile using the slate-BradleyTerry
  model.
- `slate_bt_profiles_by_bloc_generator`: Generates preference profiles by bloc using the
  slate-BradleyTerry model.
- `slate_bt_profile_generator_using_mcmc`: Generates a single preference profile using MCMC
  sampling from the slate-BradleyTerry model.
- `slate_bt_profiles_by_bloc_generator_using_mcmc`: Generates preference profiles by bloc using
  MCMC sampling from the slate-BradleyTerry model.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from votekit.ballot_generator.bloc_slate_generator._profile_construction import (
    _apportion_bloc_rows,
    _build_profile_from_rankings,
)
from votekit.ballot_generator.bloc_slate_generator._sampling import (
    BATCH_SIZE,
    _normalize_log_weights_to_cdf_in_place,
    _sample_cdf_sequences,
)
from votekit.ballot_generator.bloc_slate_generator._slate_bt_mcmc import (
    slate_bt_profile_generator_using_mcmc,
    slate_bt_profiles_by_bloc_generator_using_mcmc,
)
from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig
from votekit.ballot_generator.bloc_slate_generator.slate_utils import (
    _fill_slate_candidate_rankings,
    _prepare_slate_candidate_arrays,
)
from votekit.ballot_generator.utils import system_memory
from votekit.pref_profile import RankProfile
from votekit.types import Candidate

# Mark intentional re-export of the MCMC-based generators for public API preservation.
__all__ = [
    "slate_bt_profile_generator",
    "slate_bt_profiles_by_bloc_generator",
    "slate_bt_profile_generator_using_mcmc",
    "slate_bt_profiles_by_bloc_generator_using_mcmc",
]


# =================================================
# ================= API Functions =================
# =================================================


def slate_bt_profile_generator(
    config: BlocSlateConfig,
    *,
    group_ballots: bool = True,
    rng_seed: int | None = None,
) -> RankProfile:
    r"""
    Generate a preference profile using the slate-BradleyTerry model.

    The model separates preferences between slates from preferences among candidates within a
    slate. A complete candidate ranking determines a sequence of slate symbols, such as AABABB,
    and an ordering of the candidates belonging to each slate. Cohesion determines the
    distribution over slate sequences, while candidate supports determine the independent
    within-slate orderings.

    For the current bloc, let :math:`a_j>0` be slate :math:`j`'s cohesion and let
    :math:`\mathcal G` contain all distinct slate sequences with one occurrence per candidate in
    each slate. For :math:`m` candidates, a sequence :math:`G=(g_1,\ldots,g_m)\in\mathcal G` has
    unnormalized weight

    .. math::

        q(G)=\prod_{u<v:\,g_u\ne g_v}\frac{a_{g_u}}{a_{g_u}+a_{g_v}}.

    Only comparisons between different slates contribute to this weight. Normalizing over
    :math:`\mathcal G` gives the sequence probability:

    .. math::

        P(G)=\frac{q(G)}{\sum_{G'\in\mathcal G}q(G')}.

    For two candidates from slate A and one from slate B, the possible sequences are AAB, ABA, and
    BAA. Their weights share a denominator, leaving relative weights
    :math:`a_A^2`, :math:`a_Aa_B`, and :math:`a_B^2`. With cohesions in the ratio
    :math:`3:2`, these sequences therefore have probabilities :math:`9/19`, :math:`6/19`, and
    :math:`4/19`.

    Within each slate, candidates follow the Plackett-Luce distribution defined by their
    preference interval for the bloc. If slate :math:`j` has :math:`M_j` candidates with positive
    supports :math:`s_i`, its candidate ordering :math:`R_j=(r_1,\ldots,r_{M_j})` has probability

    .. math::

        P(R_j)=\prod_{u=1}^{M_j}\frac{s_{r_u}}{\sum_{v=u}^{M_j}s_{r_v}}.

    A complete ranking :math:`R` uniquely determines its slate sequence :math:`G(R)` and each
    within-slate ordering :math:`R_j`. These choices are independent, so its probability is
    :math:`P(R)=P(G(R))\prod_j P(R_j)`.

    When permitted by the configuration, zero-support candidates form an independently uniform
    shuffle at the end of their slate's candidate ordering. Slates with zero cohesion follow all
    positive-cohesion slates, with a uniform distribution over distinct tail sequences having the
    required slate counts. Within-slate candidate preferences still apply to those tail positions.
    Both tail policies apply before identical ballots are grouped.

    Args:
        config (BlocSlateConfig): Slate membership, bloc proportions, cohesion, candidate
            supports, and voter count.
        group_ballots (bool, optional): Combine identical complete rankings and sum their weights.
            Defaults to True.
        rng_seed (int | None, optional): Seed for NumPy's generator. Defaults to None. Seeded
            samples differ from the enumeration sampler but are reproducible for identical inputs.

    Returns:
        RankProfile: Complete rankings with the declared candidate order and total voter weight.

    Raises:
        ValueError: The configuration is invalid or array sizes exceed the platform index width.
        MemoryError: Estimated peak memory exceeds currently available RAM.
    """
    config.is_valid(raise_errors=True)
    candidates = tuple(config.candidates)
    rng = np.random.default_rng(rng_seed)
    rankings, _ = _sample_slate_bt_rankings(config, candidates, rng, group_ballots=group_ballots)

    return _build_profile_from_rankings(rankings, candidates, group_ballots=group_ballots)


def slate_bt_profiles_by_bloc_generator(
    config: BlocSlateConfig,
    *,
    group_ballots: bool = True,
    rng_seed: int | None = None,
) -> dict[str, RankProfile]:
    """
    Generate slate-BradleyTerry preference profiles separately for each bloc.

    Use the same distribution, zero-support policies, and random stream as
    ``slate_bt_profile_generator``. Both APIs sample the same rankings for a given seed. Blocs
    follow configuration order and share one seeded NumPy generator. Grouping combines identical
    complete rankings within each bloc.

    Args:
        config (BlocSlateConfig): Slate membership, bloc proportions, cohesion, supports, and
            voter count.
        group_ballots (bool, optional): Combine identical rankings within each bloc. Defaults to
            True.
        rng_seed (int | None, optional): Seed for NumPy's generator. Defaults to None.

    Returns:
        dict[str, RankProfile]: Profiles in bloc order, retaining the declared candidate universe.

    Raises:
        ValueError: The configuration is invalid or array sizes exceed the platform index width.
        MemoryError: Estimated peak memory exceeds currently available RAM.
    """
    config.is_valid(raise_errors=True)
    candidates = tuple(config.candidates)
    rng = np.random.default_rng(rng_seed)
    rankings, bloc_rows = _sample_slate_bt_rankings(
        config, candidates, rng, group_ballots=group_ballots
    )

    return {
        bloc: _build_profile_from_rankings(rankings[rows], candidates, group_ballots=group_ballots)
        for bloc, rows in bloc_rows.items()
    }


# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


def _sample_slate_bt_rankings(
    config: BlocSlateConfig,
    candidates: tuple[Candidate, ...],
    rng: np.random.Generator,
    *,
    group_ballots: bool,
) -> tuple[NDArray[np.intp], dict[str, slice]]:
    """
    Sample complete candidate rankings into consecutive rows for each apportioned bloc.

    Huntington apportionment determines the row count for each bloc. Each bloc prepares its own
    slate-count table, samples slate sequences, and fills them with candidate IDs before the next
    bloc begins. Only one CDF table is retained at a time. The returned slices let both public
    APIs construct profiles from the same samples without drawing again.

    Args:
        config (BlocSlateConfig): Validated configuration.
        candidates (tuple[Candidate, ...]): Candidate identities in output-ID order.
        rng (np.random.Generator): Random stream shared by all blocs.
        group_ballots (bool): Whether to include grouping in the memory estimate.

    Returns:
        tuple[NDArray[np.intp], dict[str, slice]]: Complete rankings and each bloc's source rows.
    """
    slates = tuple(config.slates)
    blocs = tuple(config.blocs)
    slate_counts = np.array([len(config.slate_to_candidates[slate]) for slate in slates], dtype=int)
    cohesion = config.cohesion_df.loc[list(blocs), list(slates)].to_numpy(dtype=np.float64)
    _check_dp_slate_bt_memory(slate_counts, cohesion, config.n_voters, group_ballots=group_ballots)

    bloc_rows = _apportion_bloc_rows(config)
    candidate_to_id = {candidate: index for index, candidate in enumerate(candidates)}
    rankings = np.empty((config.n_voters, len(candidates)), dtype=int)

    for (bloc, rows), bloc_cohesion in zip(bloc_rows.items(), cohesion, strict=True):
        n_bloc_voters = rows.stop - rows.start
        if n_bloc_voters == 0:
            continue

        candidate_arrays = _prepare_slate_candidate_arrays(config, bloc, candidate_to_id)
        slate_batches = _iter_slate_sequence_batches(
            slate_counts, bloc_cohesion, n_voters=n_bloc_voters, rng=rng
        )
        _fill_slate_candidate_rankings(candidate_arrays, slate_batches, rankings[rows], rng)

    return rankings, bloc_rows


def _iter_slate_sequence_batches(
    slate_counts: NDArray[np.intp],
    cohesion: NDArray[np.float64],
    n_voters: int,
    rng: np.random.Generator,
) -> Iterator[NDArray[np.intp]]:
    """
    Sample positive-cohesion slate sequences and append uniformly shuffled zero-cohesion symbols.

    A bloc's slate sequence consists of a positive-cohesion prefix followed by a zero-cohesion
    tail. Only the positive-cohesion slates enter ``_build_cdf_matrix``, so its columns use a
    local slate order. ``_sample_cdf_sequences`` draws the prefix in that order, after which the
    sampled IDs are mapped back to the configuration's full slate order. For example, if only A
    and C have positive cohesion among slates ``(A, B, C)``, their local IDs ``(0, 1)`` become
    configuration IDs ``(0, 2)``.

    The remaining positions contain one symbol per candidate from each zero-cohesion slate. An
    independent shuffle of these repeated symbols makes every distinct tail sequence equally
    likely because each sequence corresponds to the same number of permutations of the individual
    occurrences. These tails are appended before candidate identities are assigned and before
    ballots are grouped. The CDF is prepared once and retained while batches are yielded. Each
    batch owns its storage so candidate placement can finish before the next batch is sampled.
    Prefix draws, tail shuffles, and candidate placement share a stream; changing the batch size
    can therefore change seeded rankings.

    Args:
        slate_counts (NDArray[np.intp]): Candidate count for every slate in configuration order.
        cohesion (NDArray[np.float64]): Aligned, validated cohesion values for one bloc.
        n_voters (int): Number of slate sequences to draw.
        rng (np.random.Generator): Random stream advanced by prefix and tail sampling.

    Yields:
        NDArray[np.intp]: Independent batches of complete slate-ID sequences, one row per ballot.
    """
    positive_cohesion_slate_ids = np.flatnonzero(cohesion > 0)
    zero_cohesion_slate_ids = np.flatnonzero(cohesion == 0)
    positive_slate_counts = slate_counts[positive_cohesion_slate_ids]
    n_positive_candidates = int(positive_slate_counts.sum())
    n_candidates = int(slate_counts.sum())

    cdf_matrix, count_strides = _build_cdf_matrix(
        positive_slate_counts, cohesion[positive_cohesion_slate_ids]
    )
    zero_cohesion_symbols = np.repeat(
        zero_cohesion_slate_ids, slate_counts[zero_cohesion_slate_ids]
    )
    for batch_start in range(0, n_voters, BATCH_SIZE):
        n_batch_voters = min(BATCH_SIZE, n_voters - batch_start)
        batch_sequences = np.empty((n_batch_voters, n_candidates), dtype=int)
        batch_positive_sequences = batch_sequences[:, :n_positive_candidates]
        _sample_cdf_sequences(
            cdf_matrix, count_strides, batch_positive_sequences, rng, batch_size=BATCH_SIZE
        )
        np.take(
            positive_cohesion_slate_ids,
            batch_positive_sequences,
            out=batch_positive_sequences,
            mode="clip",
        )

        if len(zero_cohesion_symbols):
            tail_symbols_by_voter = np.broadcast_to(
                zero_cohesion_symbols, (len(batch_sequences), len(zero_cohesion_symbols))
            )
            rng.permuted(
                tail_symbols_by_voter,
                axis=1,
                out=batch_sequences[:, n_positive_candidates:],
            )

        yield batch_sequences


# ====================================================
# ================= Helper Functions =================
# ====================================================


def _check_dp_slate_bt_memory(
    slate_counts: NDArray[np.intp],
    cohesion: NDArray[np.float64],
    n_voters: int,
    *,
    group_ballots: bool,
) -> None:
    """
    Check count-table preparation, sampling, and profile construction against available RAM.

    The relevant memory requirement is the largest simultaneous allocation during preparation,
    sampling, or profile construction. Blocs prepare their positive-cohesion tables sequentially,
    so only one table is counted at a time alongside the retained ranking arrays. Sampling also
    needs temporary slate sequences, within-slate candidate orders, and batch buffers. Profile
    construction requires owned copies, and optional grouping is estimated with every row
    distinct.

    These estimates are compared with currently available RAM before the large arrays are
    allocated. The check can reject a request whose expected peak already exceeds that amount, but
    it cannot reserve memory against subsequent allocations by other processes.

    Args:
        slate_counts (NDArray[np.intp]): Candidate counts for all slates.
        cohesion (NDArray[np.float64]): Bloc-by-slate cohesion values in matching order.
        n_voters (int): Total output rows across blocs.
        group_ballots (bool): Whether to include grouping buffers and the grouped profile.

    Raises:
        ValueError: Array byte sizes exceed the platform index width.
        MemoryError: The estimated peak exceeds currently available RAM.
    """
    index_bytes = np.dtype(np.intp).itemsize
    float_bytes = np.dtype(np.float64).itemsize
    n_candidates = sum(int(count) for count in slate_counts)
    ranking_bytes = n_voters * n_candidates * index_bytes
    max_array_bytes = np.iinfo(np.intp).max

    if ranking_bytes > max_array_bytes:
        raise ValueError("Slate-BT ranking storage exceeds the platform array-index width.")

    preparation_peak_bytes = 0
    sampling_peak_bytes = 0
    largest_cdf_bytes = 0
    n_batch_voters = min(n_voters, BATCH_SIZE)
    slate_sequence_bytes = n_batch_voters * n_candidates * index_bytes
    retained_ranking_bytes = ranking_bytes + slate_sequence_bytes

    for bloc_cohesion in cohesion:
        positive_slate_counts = slate_counts[bloc_cohesion > 0]
        n_slates = len(positive_slate_counts)
        n_states = math.prod(int(count) + 1 for count in positive_slate_counts)
        cdf_bytes = n_states * n_slates * float_bytes
        largest_cdf_bytes = max(largest_cdf_bytes, cdf_bytes)

        if cdf_bytes > max_array_bytes:
            raise ValueError("Slate-BT count table exceeds the platform array-index width.")

        # Log contributions, log totals, and count-state indexing temporaries.
        preparation_rows = min(n_states, BATCH_SIZE)
        preparation_work_bytes = n_states * (float_bytes + 4 * index_bytes) + preparation_rows * (
            n_slates * float_bytes + 10 * index_bytes
        )
        bloc_preparation_bytes = retained_ranking_bytes + cdf_bytes + preparation_work_bytes
        preparation_peak_bytes = max(preparation_peak_bytes, bloc_preparation_bytes)

        cdf_and_comparison_row_bytes = n_slates * (float_bytes + 1)
        sampling_buffer_bytes = n_batch_voters * (cdf_and_comparison_row_bytes + 3 * index_bytes)
        voter_state_bytes = n_batch_voters * index_bytes
        bloc_sampling_bytes = (
            retained_ranking_bytes + cdf_bytes + voter_state_bytes + sampling_buffer_bytes
        )
        sampling_peak_bytes = max(sampling_peak_bytes, bloc_sampling_bytes)

    # Slate positions, candidate permutations, exponential keys, and indexed copies.
    candidate_ordering_work_bytes = n_batch_voters * n_candidates * (8 * index_bytes + 1)
    ordering_peak_bytes = retained_ranking_bytes + largest_cdf_bytes + candidate_ordering_work_bytes

    profile_row_metadata_bytes = n_voters * (4 * index_bytes + 4 * float_bytes)
    construction_peak_bytes = 4 * ranking_bytes + profile_row_metadata_bytes
    if group_ballots:
        grouping_row_metadata_bytes = n_voters * (12 * index_bytes + 4 * float_bytes)
        construction_peak_bytes = 5 * ranking_bytes + grouping_row_metadata_bytes

    estimated_peak_bytes = max(
        preparation_peak_bytes,
        sampling_peak_bytes,
        ordering_peak_bytes,
        construction_peak_bytes,
    )
    available_gib = system_memory()["available_gib"]

    if estimated_peak_bytes > available_gib * 2**30:
        raise MemoryError(
            f"Not enough memory for DP slate-BradleyTerry. Estimated peak is "
            f"{estimated_peak_bytes / 2**30:.3f} GiB; only {available_gib:.3f} GiB is available. "
            "Reduce slate sizes or voter count, or use the MCMC generator."
        )


def _build_cdf_matrix(
    slate_counts: NDArray[np.intp], cohesion: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    r"""
    Build conditional slate-choice CDFs for every vector of remaining slate counts.

    Sampling a slate sequence one position at a time requires the probability of choosing each
    slate next, given how many of its candidates remain. Candidate identities do not affect this
    choice: two prefixes that leave the same counts have the same distribution over subsequent
    slate symbols. The table prepares that distribution for every possible remaining-count vector.

    Let :math:`d` be the number of positive-cohesion slates, with cohesions :math:`a_j>0`, and let
    :math:`\mathbf c=(c_0,\ldots,c_{d-1})` contain their remaining counts. A suffix has
    :math:`k=\sum_j c_j` symbols. Every possible suffix with those counts has the same denominator
    in the pairwise BT formula, so normalization cancels that denominator. Each comparison between
    different slates then contributes only the cohesion of whichever slate appears first. Write
    :math:`Z(\mathbf c)` for the sum of these reduced weights over all distinct suffix sequences.

    To find this total without enumerating the suffixes, group them by their first symbol. A
    suffix beginning with slate :math:`j` places that symbol ahead of :math:`k-c_j` symbols from
    other slates, contributing :math:`a_j^{k-c_j}`. The remaining factors form the weight of a
    suffix with counts :math:`\mathbf c-\mathbf e_j`, where :math:`\mathbf e_j` has a one in
    position :math:`j` and zeros elsewhere. Summing first within each group and then across
    possible first symbols gives

    .. math::

        Z(\mathbf 0)=1,\qquad
        Z(\mathbf c)=\sum_{j:c_j>0}a_j^{k-c_j}Z(\mathbf c-\mathbf e_j).

    No factor :math:`c_j` is needed because these branches distinguish slate symbols, not
    candidate identities. For counts corresponding to AAB, there is one branch beginning with A,
    whose continuations are AB and BA. Treating the two A occurrences as separate choices would
    count those same sequences twice. Candidate identities receive their own orderings later.

    Every term uses a total for one fewer remaining symbol, which determines the order of
    preparation. The empty suffix has one ordering with weight one. A suffix containing only one
    slate also has weight one because it contains no comparisons between different slates.
    Starting from these cases, the implementation processes states by increasing total count. For
    two A symbols and one B symbol with cohesions :math:`a_A=3` and :math:`a_B=2`, it first
    obtains :math:`Z(1,1)=3+2=5` and :math:`Z(2,0)=1`, then combines them:

    .. math::

        Z(2,1)=3^1Z(1,1)+2^2Z(2,0)=15+4=19.

    The first contribution accounts for AAB and ABA, whose reduced weights are nine and six. The
    second accounts for BAA, whose weight is four. Dividing each contribution by their total gives
    first-choice probabilities :math:`15/19` for A and :math:`4/19` for B. The same reasoning
    applies after any fixed prefix: its internal comparisons and comparisons with the remaining
    symbols contribute a common factor to every continuation. That factor cancels, leaving

    .. math::

        p(j\mid\mathbf c)=\frac{a_j^{k-c_j}Z(\mathbf c-\mathbf e_j)}{Z(\mathbf c)}.

    Although the resulting probabilities may be ordinary-sized numbers, the products used to
    obtain them can underflow when cohesions are small or overflow when unnormalized cohesions are
    large. Computing their logarithms preserves the information needed to form the ratios without
    constructing those products. ``log_total_suffix_weight`` stores
    :math:`L(\mathbf c)=\log Z(\mathbf c)`, and ``log_next_slate_weights`` holds each available
    slate's contribution :math:`(k-c_j)\log a_j+L(\mathbf c-\mathbf e_j)`. Log-sum-exp combines
    these contributions into :math:`L(\mathbf c)`. Subtracting that total and exponentiating
    produces the conditional probabilities. Exhausted slates have log contribution minus infinity
    and therefore probability zero.

    These totals and probabilities need one array row for each count vector. With initial counts
    :math:`\mathbf M=(M_0,\ldots,M_{d-1})`, slate :math:`j` can have any count from zero through
    :math:`M_j`, giving it :math:`M_j+1` possible values. Treating each count as a digit encodes
    the vector as a single integer. Unlike decimal notation, the digits can have different bases,
    so this is a mixed-radix encoding. ``count_radices`` contains those bases, while
    ``count_strides`` contains the place value of each digit:

    .. math::

        h_0=1,\qquad h_j=\prod_{i<j}(M_i+1),\qquad
        b(\mathbf c)=\sum_j c_jh_j.

    For the AAB example, A's count has three possible values and B's has two. Their strides are
    therefore :math:`(1,3)`, and :math:`b(c_A,c_B)=c_A+3c_B`. The six array rows represent

    .. math::

        \begin{array}{c|rrrrrr}
        b(\mathbf c) & 0 & 1 & 2 & 3 & 4 & 5 \\
        \mathbf c & (0,0) & (1,0) & (2,0) & (0,1) & (1,1) & (2,1)
        \end{array}

    The initial counts occupy row five. Choosing A subtracts its stride of one, reaching row four
    for counts :math:`(1,1)`; choosing B subtracts three, reaching row two for counts
    :math:`(2,0)`. Both preparation and sampling can therefore locate the next state by subtracting
    the chosen slate's stride. Row zero represents the empty suffix and is never sampled.

    Each nonempty row stores cumulative conditional probabilities in slate order. In the example,
    row five is :math:`(15/19,1)` and row four is :math:`(3/5,1)`. An exhausted slate contributes
    no probability, so its column leaves the cumulative value unchanged. Every row should end at
    one because it includes all possible next choices. Dividing by the final cumulative value
    corrects small floating point accumulation errors before the row is stored.

    There are :math:`N=\prod_j(M_j+1)` states and :math:`d` columns, so the retained float64 table
    occupies :math:`8dN` bytes in addition to the strides and temporary preparation buffers.
    Transition calculations take :math:`O(dN)` work. Selecting states by total count also scans
    the state vector once for each of the :math:`m=\sum_j M_j` positions, taking :math:`O(mN)`
    additional work.

    Args:
        slate_counts (NDArray[np.intp]): Positive candidate counts in local slate-ID order.
        cohesion (NDArray[np.float64]): Aligned finite, positive cohesion values. The caller
            validates inputs and checks memory before allocating the table.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.intp]]: An ``(N, d)`` conditional CDF table and the
            stride to subtract when each slate is chosen.
    """
    n_slates = len(slate_counts)
    n_candidates = int(slate_counts.sum())
    count_radices = slate_counts + 1
    count_strides = np.ones(n_slates, dtype=int)
    np.cumprod(count_radices[:-1], out=count_strides[1:])

    n_states = math.prod(int(radix) for radix in count_radices)
    count_state_ids = np.arange(n_states, dtype=int)
    n_candidates_by_state = np.zeros(n_states, dtype=int)

    for count_radix, count_stride in zip(count_radices, count_strides, strict=True):
        n_candidates_by_state += (count_state_ids // count_stride) % count_radix

    log_cohesion = np.log(cohesion)
    log_total_suffix_weight = np.full(n_states, -np.inf)

    # The empty suffix has one ordering with weight one.
    log_total_suffix_weight[0] = 0

    cdf_matrix = np.zeros((n_states, n_slates), dtype=np.float64)

    for n_remaining_candidates in range(1, n_candidates + 1):
        states_with_n_remaining_candidates = count_state_ids[
            n_candidates_by_state == n_remaining_candidates
        ]
        for start in range(0, len(states_with_n_remaining_candidates), BATCH_SIZE):
            selected_states = states_with_n_remaining_candidates[start : start + BATCH_SIZE]
            _fill_slate_bt_cdf_rows(
                count_state_ids=selected_states,
                n_remaining_candidates=n_remaining_candidates,
                count_radices=count_radices,
                count_strides=count_strides,
                log_cohesion=log_cohesion,
                log_total_suffix_weight=log_total_suffix_weight,
                cdf_matrix=cdf_matrix,
            )

    return cdf_matrix, count_strides


def _fill_slate_bt_cdf_rows(
    count_state_ids: NDArray[np.intp],
    n_remaining_candidates: int,
    count_radices: NDArray[np.intp],
    count_strides: NDArray[np.intp],
    log_cohesion: NDArray[np.float64],
    log_total_suffix_weight: NDArray[np.float64],
    cdf_matrix: NDArray[np.float64],
) -> None:
    """
    Compute and store suffix totals and next-slate CDFs for a batch of count states.

    Choosing a slate first contributes its cohesion once for every remaining candidate in another
    slate. Multiplying that contribution by the successor state's completion weight accounts for
    all suffix sequences beginning with that slate. Each choice removes one occurrence, so its
    successor has one fewer remaining candidate and must already have been computed. No factor for
    the chosen slate's remaining count is needed: the choices are distinct slate symbols, and
    candidate identities are assigned after sampling the sequence.

    The function evaluates these contributions in log space and stores their log total for
    subsequent states. It then reuses the temporary choice-weight array to normalize and
    accumulate the next-slate probabilities. Only the selected entries in the supplied
    total-weight vector and CDF matrix are modified, and the temporary array is released when the
    function returns.

    Args:
        count_state_ids (NDArray[np.intp]): Nonempty mixed-radix count states identifying the rows
            to fill. Each state contains ``n_remaining_candidates`` remaining slate occurrences.
        n_remaining_candidates (int): Total number of candidate positions remaining in each state.
        count_radices (NDArray[np.intp]): Initial slate counts plus one, in CDF column order.
        count_strides (NDArray[np.intp]): Products of preceding radices in the same order.
            Subtracting a slate's stride removes one occurrence from a state where it is present.
        log_cohesion (NDArray[np.float64]): Logarithms of finite, positive slate cohesions in CDF
            column order.
        log_total_suffix_weight (NDArray[np.float64]): Writable log completion totals indexed by
            count state. Successor entries must be initialized, including zero for the empty
            state. Selected entries are filled in place.
        cdf_matrix (NDArray[np.float64]): Writable table indexed by count state and slate ID.
            Selected rows are filled in place, with zero probability for exhausted slates and a
            final cumulative probability of one.
    """
    n_slates = len(count_strides)
    n_selected_states = len(count_state_ids)
    log_next_slate_weights = np.full((n_selected_states, n_slates), -np.inf)

    for slate_id, count_stride in enumerate(count_strides):
        remaining_slate_counts = (count_state_ids // count_stride) % count_radices[slate_id]
        slate_is_available = remaining_slate_counts > 0

        successor_state_ids = count_state_ids[slate_is_available] - count_stride
        n_candidates_in_other_slates = (
            n_remaining_candidates - remaining_slate_counts[slate_is_available]
        )
        log_next_slate_weights[slate_is_available, slate_id] = (
            n_candidates_in_other_slates * log_cohesion[slate_id]
            + log_total_suffix_weight[successor_state_ids]
        )

    log_state_totals = _normalize_log_weights_to_cdf_in_place(log_next_slate_weights)

    log_total_suffix_weight[count_state_ids] = log_state_totals
    cdf_matrix[count_state_ids] = log_next_slate_weights
