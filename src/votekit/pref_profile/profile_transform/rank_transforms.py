from dataclasses import dataclass, replace

import numpy as np

from votekit.constants import (
    EMPTY_SET_ID,
    END_OF_RANKING_ID,
)
from votekit.pref_profile.profile_matrix import RankMatrix
from votekit.pref_profile.profile_matrix.shared_data import _concatenate_voter_weight_data
from votekit.pref_profile.profile_transform.transform_utils import (
    _int_weights,
    _validate_one_voter_per_unit_weight,
)
from votekit.types import Candidate


@dataclass
class SwapCandidates:
    """
    ProfileTransform class to swap two candidates on every ballot.

    Every candidate set holding one of the two candidates has it replaced by the other, so a tie
    with one becomes a tie with the other. A set holding both, or neither, is unchanged. The swap
    is a mapping over the matrix's candidate-set table, applied to all ballots at once,
    regardless of the candidates' distance or order. Swapped sets that are not already in the
    table are appended to it, and sets that are no longer used stay in the table.

    Attributes:
        candidate_a (Candidate): Candidate to swap with candidate_b.
        candidate_b (Candidate): Candidate to swap with candidate_a.

    Raises:
        ValueError: candidate_a or candidate_b is not one of the profile's candidates.
        ValueError: A ballot ranks candidate_a or candidate_b in more than one position, so the
            swap is ambiguous.
    """

    candidate_a: Candidate
    candidate_b: Candidate

    def transform_matrix(self, matrix: RankMatrix, rng: np.random.Generator) -> RankMatrix:
        undeclared_cands = {self.candidate_a, self.candidate_b} - set(matrix.candidates)
        if undeclared_cands:
            raise ValueError(
                f"SwapCandidates got candidates {undeclared_cands} not in the profile's"
                f" candidates {matrix.candidates}. Declare them before transforming."
            )

        cand_sets = matrix.cand_sets
        has_a = np.fromiter((self.candidate_a in s for s in cand_sets), bool, len(cand_sets))
        has_b = np.fromiter((self.candidate_b in s for s in cand_sets), bool, len(cand_sets))

        # fail before swapping if a ballot ranks either candidate in more than one position
        slots = matrix.rankings - EMPTY_SET_ID
        repeated = (np.count_nonzero(has_a[slots], axis=1) > 1) | (
            np.count_nonzero(has_b[slots], axis=1) > 1
        )
        if np.any(repeated):
            row_idx = int(np.flatnonzero(repeated)[0])
            raise ValueError(
                f"Cannot swap {self.candidate_a} and {self.candidate_b}: the ballot"
                f" {matrix.get_ranking(row_idx)} ranks one of them in more than one position."
            )

        # identity map from each candidate-set ID to its swapped ID; only sets holding exactly
        # one of the two candidates change, and new swapped sets are appended to the table
        id_mapping = np.arange(EMPTY_SET_ID, EMPTY_SET_ID + len(cand_sets), dtype=np.intp)
        cand_set_to_id = {
            cand_set: cand_set_id for cand_set_id, cand_set in matrix.iter_candidate_sets()
        }
        swap = {self.candidate_a: self.candidate_b, self.candidate_b: self.candidate_a}
        for slot in np.flatnonzero(has_a != has_b):
            swapped_set = frozenset(swap.get(candidate, candidate) for candidate in cand_sets[slot])
            id_mapping[slot] = cand_set_to_id.setdefault(
                swapped_set, len(cand_set_to_id) + EMPTY_SET_ID
            )

        return replace(
            matrix,
            rankings=id_mapping[slots],
            cand_sets=tuple(cand_set_to_id),  # insertion order is ID order
        )


@dataclass
class SwapAdjacentRankPositions:
    """
    ProfileTransform class to swap one random pair of adjacent ranking positions for each voter.

    For each voter, a position is drawn uniformly from the profile's ranking positions and a
    direction, left or right, is drawn uniformly. The ballot's candidate sets at that position and
    its neighbor are swapped. If the neighbor falls off either end of the ranking positions, or
    either position is unranked padding, the ballot is unchanged. An empty ranking position is
    swapped like any other position.

    Rather than drawing for each voter, one multinomial per row draws how many of its voters make
    each of the 2m moves (a position and a direction) for m ranking positions. The moves are
    binned by the adjacent pair they swap, and moves off either end or into padding are no-ops.
    Each pair's voters are binned out of the row in turn, subtracting them from the voters left,
    and the voters left at the end keep their ballot. A row's voters are assigned to its outcomes
    at random.

    Use with ``transform``, whose ``per_voter_probability`` selects the voters to swap. The swaps
    are drawn from the generator ``transform`` passes in, so its ``rng`` reproduces them.

    Raises:
        ValueError: The matrix has non-integral weights, so its voters cannot be counted.
        ProfileError: A row's non-empty voter set does not have one voter per unit of weight.
    """

    def transform_matrix(self, matrix: RankMatrix, rng: np.random.Generator) -> RankMatrix:
        _validate_one_voter_per_unit_weight(matrix)
        num_positions = matrix.num_ranking_positions
        if len(matrix) == 0 or num_positions < 2:
            return matrix

        weights = _int_weights(matrix.voter_weight_data)
        move_probabilities = np.full(2 * num_positions, 1 / (2 * num_positions))
        moves = rng.multinomial(weights, move_probabilities).reshape(len(matrix), num_positions, 2)
        moves_left, moves_right = moves[:, :, 0], moves[:, :, 1]

        swap_counts = moves_right[:, :-1] + moves_left[:, 1:]
        is_ranked = matrix.rankings != END_OF_RANKING_ID
        swap_counts[~(is_ranked[:, :-1] & is_ranked[:, 1:])] = 0

        unswapped_matrix = matrix
        swapped_rankings, swapped_voter_data = [], []
        for left in range(num_positions - 1):
            pair_counts = swap_counts[:, left]
            swapped_matrix, unswapped_matrix = unswapped_matrix._apply_bin_mask(
                pair_counts, rng=rng
            )

            swapped_columns = np.arange(num_positions)
            swapped_columns[[left, left + 1]] = [left + 1, left]
            swapped_rankings.append(swapped_matrix.rankings[:, swapped_columns])
            swapped_voter_data.append(swapped_matrix.voter_weight_data)

            weights = weights - pair_counts
            still_unswapped = (weights > 0) | (pair_counts == 0)
            weights, swap_counts = weights[still_unswapped], swap_counts[still_unswapped]

        return RankMatrix(
            voter_weight_data=_concatenate_voter_weight_data(
                [unswapped_matrix.voter_weight_data, *swapped_voter_data]
            ),
            rankings=np.concatenate([unswapped_matrix.rankings, *swapped_rankings]),
            cand_sets=matrix.cand_sets,
            ranking_columns=matrix.ranking_columns,
            candidates=matrix.candidates,
        )
