from dataclasses import dataclass, field, replace

import numpy as np

from votekit.constants import (
    EMPTY_SET_ID,
    END_OF_RANKING_ID,
)
from votekit.pref_profile.profile_matrix import RankMatrix
from votekit.pref_profile.profile_transform.transform_utils import (
    _ballot_label,
    _expand_to_unit_weights,
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

    def transform_matrix(self, matrix: RankMatrix) -> RankMatrix:
        undeclared_cands = {self.candidate_a, self.candidate_b} - set(matrix.candidates)
        if undeclared_cands:
            raise ValueError(
                f"SwapCandidates got candidates {undeclared_cands} not in the profile's"
                f" candidates {matrix.candidates}. Declare them before perturbing."
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
                f"Cannot swap {self.candidate_a} and {self.candidate_b}: Ballot Index"
                f" {_ballot_label(matrix.voter_weight_data, row_idx)} ranks one of them in more"
                " than one position."
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
    ProfileTransform class to swap one random pair of adjacent ranking positions on each ballot.

    For each ballot, a position is drawn uniformly from the profile's ranking positions and a
    direction, left or right, is drawn uniformly. The ballot's candidate sets at that position and
    its neighbor are swapped. If the neighbor falls off either end of the ranking positions, or
    either position is unranked padding, the ballot is unchanged. An empty ranking position is
    swapped like any other position.

    Use with ``perturb``, whose ``per_voter_probability`` selects the ballots to jitter. Each
    selected voter receives its own swap, so rows are split into one row per unit of weight.

    The generator is created once from ``rng_seed``, so repeated calls, such as repeated
    ``perturb`` runs with the same transform, continue its random stream instead of repeating
    the same swaps.

    Attributes:
        rng_seed (int, optional): Seed for sampling the positions and directions. Pass a seed for
            reproducible swaps. Defaults to None, which creates an unseeded generator.
    """

    rng_seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.rng_seed)

    def transform_matrix(self, matrix: RankMatrix) -> RankMatrix:
        unit_rows, unit_data = _expand_to_unit_weights(matrix)
        ranking_arr = matrix.rankings[unit_rows]
        num_rows, num_positions = ranking_arr.shape

        row_idxs = np.arange(num_rows)
        positions = self._rng.integers(0, max(num_positions, 1), size=num_rows)
        neighbors = positions + self._rng.choice([-1, 1], size=num_rows)

        in_range = (neighbors >= 0) & (neighbors < num_positions)
        row_idxs, positions, neighbors = (
            row_idxs[in_range],
            positions[in_range],
            neighbors[in_range],
        )

        ranked = (ranking_arr[row_idxs, positions] != END_OF_RANKING_ID) & (
            ranking_arr[row_idxs, neighbors] != END_OF_RANKING_ID
        )
        row_idxs, positions, neighbors = row_idxs[ranked], positions[ranked], neighbors[ranked]

        ranking_arr[row_idxs, positions], ranking_arr[row_idxs, neighbors] = (
            ranking_arr[row_idxs, neighbors],
            ranking_arr[row_idxs, positions],
        )

        return replace(matrix, voter_weight_data=unit_data, rankings=ranking_arr)
