"""
Sample weighted candidate rankings and sequences from prepared conditional distributions.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

SAMPLING_BATCH_SIZE = 65_536


def _sample_pl_rankings(
    candidate_ids: NDArray[np.intp],
    candidate_supports: NDArray[np.float64],
    *,
    n_voters: int,
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    r"""
    Sample Plackett-Luce rankings followed by uniformly shuffled zero-support tails.

    Positive-support candidates follow the weighted ordering sampled by
    ``_fast_sample_without_replacement``. Its returned indices are mapped to the supplied
    candidate IDs. Zero-support candidates are shuffled independently and placed after the
    positive-support candidates in each ranking.

    Args:
        candidate_ids (NDArray[np.intp]): Candidate IDs in support-vector order.
        candidate_supports (NDArray[np.float64]): Aligned, validated finite nonnegative supports.
        n_voters (int): Number of independent candidate orderings to draw.
        rng (np.random.Generator): Random stream advanced by exponential draws, then tail
            shuffles.

    Returns:
        NDArray[np.intp]: One complete ordering of the supplied candidate IDs per voter.
    """
    positive_support_mask = candidate_supports > 0
    positive_candidate_ids = candidate_ids[positive_support_mask]
    zero_support_candidate_ids = candidate_ids[candidate_supports == 0]
    n_positive_candidates = len(positive_candidate_ids)

    candidate_orderings = np.empty((n_voters, len(candidate_ids)), dtype=int)
    arrival_order = _fast_sample_without_replacement(
        candidate_supports[positive_support_mask], n_voters, numpy_rng=rng
    )
    candidate_orderings[:, :n_positive_candidates] = positive_candidate_ids[arrival_order]

    if len(zero_support_candidate_ids):
        tail_candidates_by_voter = np.broadcast_to(
            zero_support_candidate_ids, (n_voters, len(zero_support_candidate_ids))
        )
        rng.permuted(
            tail_candidates_by_voter,
            axis=1,
            out=candidate_orderings[:, n_positive_candidates:],
        )

    return candidate_orderings


def _fast_sample_without_replacement(
    weights: NDArray[np.float64],
    n_samples: int,
    *,
    numpy_rng: Generator | None = None,
) -> NDArray[np.intp]:
    r"""
    Sample complete weighted rankings without replacement, returning candidate indices.

    Each candidate receives an independent exponential arrival time with rate equal to its weight.
    The probability that candidate :math:`i` arrives first is :math:`w_i/\sum_j w_j`. After that
    arrival, the memoryless property gives the same competition among the remaining candidates.
    Sorting the arrival times therefore produces the Plackett-Luce distribution: each next
    candidate is selected in proportion to its weight among those still available.

    For unit-rate exponential draws :math:`E_i`, the arrival times are :math:`E_i/w_i`. Sorting
    :math:`\log E_i-\log w_i` preserves their order without dividing by weights that may be very
    small. An exact zero draw gives log arrival time :math:`-\infty` and places that candidate
    first. Zero weights are excluded by the caller and handled separately as an unweighted tail.

    This construction is equivalent to Algorithm A's weighted ordering in
    https://doi.org/10.1016/j.ipl.2005.11.003: writing :math:`E_i=-\log U_i` shows that sorting
    :math:`E_i/w_i` increasingly gives the same order as sorting :math:`U_i^{1/w_i}` decreasingly.
    The logarithmic form avoids the underflow that can collapse those powers to tied zeros.

    Args:
        weights (NDArray[np.float64]): One-dimensional, finite, strictly positive candidate
            weights. They need not sum to one. The caller is expected to validate these
            conditions.
        n_samples (int): Number of independent complete rankings to generate.
        numpy_rng (Generator, optional): NumPy random number generator. Pass a seeded instance for
            reproducible results. Defaults to None for non-deterministic results.

    Returns:
        NDArray[np.intp]: An ``(n_samples, len(weights))`` array with each row containing every
            candidate index once, in sampled preference order.
    """
    n_cands = len(weights)
    numpy_rng = np.random.default_rng() if numpy_rng is None else numpy_rng
    log_arrival_times = numpy_rng.standard_exponential(size=(n_samples, n_cands))
    with np.errstate(divide="ignore"):
        np.log(log_arrival_times, out=log_arrival_times)

    log_arrival_times -= np.log(weights)
    return np.argsort(log_arrival_times, axis=1)


def _sample_cdf_sequences(
    cdf_matrix: NDArray[np.float64],
    state_decrements: NDArray[np.intp],
    sequences: NDArray[np.intp],
    rng: np.random.Generator,
    *,
    batch_size: int,
) -> None:
    r"""
    Fill sequences by sampling successive choices from their conditional CDFs.

    A prepared CDF table gives the next-choice distribution for every possible remaining state.
    Sampling a complete sequence therefore requires only a current state and a sequence of random
    draws: choose an available candidate or slate, update what remains, and continue from the
    resulting state's CDF. The table supplies the probabilities; ``state_decrements`` supplies the
    change in state index associated with each choice. For :math:`n` voters and :math:`m`
    positions, the function writes the chosen CDF column IDs into the supplied :math:`(n,m)`
    array.

    Consider one voter whose remaining state has index :math:`b`. Its CDF row divides
    :math:`[0,1)` into intervals with lengths equal to the next-choice probabilities. A uniform
    draw selects an interval and therefore its candidate or slate. More precisely, for
    :math:`u\sim\operatorname{Uniform}[0,1)`, the chosen column is the first whose cumulative
    probability exceeds the draw:

    .. math::

        j=\min\{i:u<\mathrm{cdf}[b,i]\}.

    For example, name-BT with candidate order ``(A, B, C)`` and supports :math:`(3,2,1)` has
    first-choice weights :math:`(27,16,5)`. All three candidates are initially available, so the
    subset bitmask is seven (``0b111``) and its CDF row is

    .. math::

        \mathrm{cdf}[7,:]=\left(\frac{27}{48},\frac{43}{48},1\right).

    A draw of :math:`0.7` falls between the first two thresholds and selects B, whose ID is one.
    For candidate subsets, ``state_decrements`` contains the candidate bits :math:`(1,2,4)`.
    Subtracting B's bit changes the state from seven to :math:`7-2=5`, leaving A and C. Their
    next-choice probabilities are :math:`3/4` and :math:`1/4`, but the row still has columns for
    all three candidates:

    .. math::

        \mathrm{cdf}[5,:]=\left(\frac34,\frac34,1\right).

    B's repeated threshold gives it an interval of length zero because it is no longer available.
    A new draw of :math:`0.6` selects A and changes the state to :math:`5-1=4`. Only C remains, so
    the final choice completes the encoded ranking ``(1, 0, 2)``, or :math:`B>A>C`. Every choice
    removes a present bit, ensuring that a candidate appears exactly once. After the final choice
    the state is zero and no further lookup is needed.

    Slate sequences use the same loop with a different encoding. A state records the number of
    occurrences remaining for each slate, and its decrements are the mixed-radix strides used to
    encode those counts. With initial counts :math:`(2,1)`, the strides are :math:`(1,3)` and the
    state index is :math:`2+3=5`. Selecting A subtracts one, giving index four for counts
    :math:`(1,1)`. Selecting B next subtracts three, leaving index one for counts :math:`(1,0)`.
    A final A reaches zero and completes ABA. Unlike a candidate bit, a slate's stride can be
    subtracted repeatedly until its count reaches zero; the CDF then makes that slate unavailable.

    These conditional draws reproduce the distribution encoded by the table. In the BT tables,
    each choice's completion weight accounts for all possible continuations. For a name-BT ranking
    :math:`R=(r_1,\ldots,r_m)\in S_C`, where :math:`S_C` is the symmetric group on candidate set
    :math:`C`, let :math:`U_t=\{r_t,\ldots,r_m\}` be the subset remaining before position
    :math:`t`, with :math:`U_{m+1}=\varnothing`. Writing :math:`s_i` for candidate support and
    :math:`Z(U)` for the total suffix weight on :math:`U`, multiplication gives

    .. math::

        P(R)
        =\prod_{t=1}^{m}\frac{s_{r_t}^{m-t}Z(U_{t+1})}{Z(U_t)}
        =\frac{\prod_{t=1}^{m}s_{r_t}^{m-t}}{Z(C)}.

    Each numerator's suffix total cancels the next denominator, leaving only :math:`Z(C)` below
    and :math:`Z(\varnothing)=1` above. In the example, the probability of B, then A, then C is
    :math:`(16/48)(3/4)(1)=12/48`, matching its reduced ranking weight :math:`s_B^2s_A=12`
    divided by :math:`Z(C)=48`. Slate-BT has the same cancellation over remaining-count states.
    These identities describe the prepared distributions; sampling uses their stored float64
    approximations.

    To select a column, we need the index of the first CDF entry greater than the random draw. We
    can find that index by counting how many entries precede it. Every preceding entry is less
    than or equal to the draw, so comparing the draw with the entire row and counting the
    successful comparisons gives the selected column's index. For :math:`q` CDF columns,

    .. math::

        j=\sum_{i=0}^{q-1}\mathbf{1}\{\mathrm{cdf}[b,i]\leq u\}.

    For example, the CDF row :math:`(27/48,43/48,1)` and a draw of :math:`0.7` produce
    ``(True, False, False)``. Only the first threshold is at or below the draw, so the selected
    column is one, corresponding to B. NumPy performs these comparisons for every voter in a batch
    and sums each row to obtain one choice per voter.

    Repeated thresholds fit the same rule. After B has been selected, the remaining set
    :math:`\{A,C\}` has CDF row :math:`(3/4,3/4,1)`. A occupies :math:`[0,3/4)`, and C
    occupies :math:`[3/4,1)`; B has no interval because its probability is zero. At a draw of
    exactly :math:`3/4`, the comparisons are ``(True, True, False)``, selecting column two, or C.
    Counting entries equal to the draw moves past the entire repeated threshold and skips every
    zero-probability choice. Since the final threshold is one and draws are strictly below one,
    the selected column also stays within the table.

    Each voter needs a separate CDF lookup because earlier choices may leave different states.
    Within a batch, ``voter_cdf_rows`` holds one such row per voter, and ``draws[:, None]`` aligns
    each voter's draw with every threshold in that row. Comparing and summing across columns
    produces one ID per voter. The selected IDs are written to the current output position, and
    subtracting their entries in ``state_decrements`` prepares each voter's state for the next
    position.

    Batching limits the temporary CDF and comparison arrays without changing which draws reach
    which voters. The outer loop visits positions, and the inner loop visits consecutive voter
    batches, so all voters receive their first draw before any receives a second one. Changing the
    batch size preserves this assignment and therefore the sampled sequences for a fixed random
    stream. Every position consumes a draw, including forced choices. Apart from the shared table
    and output, storage consists of one state index per voter and reusable buffers for :math:`B`
    voters, using :math:`O(n+Bq)` auxiliary storage. Comparing all :math:`q` thresholds at each of
    :math:`m` positions takes :math:`O(nmq)` work. Name-BT has :math:`q=m` candidate columns;
    slate-BT has :math:`q=d` columns for its :math:`d` positive-cohesion slates.

    Args:
        cdf_matrix (NDArray[np.float64]): Prepared conditional CDF table. The final row represents
            the initial state; row zero is unused. Nonempty-state rows must be nondecreasing and
            end at one, with zero probability for unavailable choices.
        state_decrements (NDArray[np.intp]): Positive state decrement for each CDF column. Every
            selectable column must lead to a valid state, reaching zero after the final position.
            These conditions and the CDF properties are assumed rather than validated here.
        sequences (NDArray[np.intp]): Writable ``(n, m)`` output, filled in place with zero-based
            CDF column IDs. May be a view into a larger array. Its width must equal the number of
            choices needed to reach state zero from the initial state.
        rng (np.random.Generator): Generator advanced by ``n*m`` draws in position order, then
            voter order.
        batch_size (int): Positive maximum number of voters processed at once.
    """
    n_voters, n_positions = sequences.shape
    buffer_rows = min(batch_size, n_voters)
    n_choices = len(state_decrements)
    remaining_state_ids = np.full(n_voters, len(cdf_matrix) - 1, dtype=int)

    voter_cdf_rows = np.empty((buffer_rows, n_choices), dtype=np.float64)
    passed_thresholds = np.empty(voter_cdf_rows.shape, dtype=bool)
    chosen_ids = np.empty(buffer_rows, dtype=int)

    for position in range(n_positions):
        for start in range(0, n_voters, batch_size):
            stop = min(start + batch_size, n_voters)
            n_batch_voters = stop - start
            batch_state_ids = remaining_state_ids[start:stop]
            batch_cdf_rows = voter_cdf_rows[:n_batch_voters]
            batch_passed_thresholds = passed_thresholds[:n_batch_voters]
            batch_chosen_ids = chosen_ids[:n_batch_voters]

            # Valid transitions keep indices in bounds; clip avoids NumPy's raise-mode buffer.
            np.take(cdf_matrix, batch_state_ids, axis=0, out=batch_cdf_rows, mode="clip")
            draws = rng.random(n_batch_voters)
            np.greater_equal(draws[:, None], batch_cdf_rows, out=batch_passed_thresholds)
            np.sum(batch_passed_thresholds, axis=1, dtype=int, out=batch_chosen_ids)

            sequences[start:stop, position] = batch_chosen_ids
            batch_state_ids -= state_decrements[batch_chosen_ids]
