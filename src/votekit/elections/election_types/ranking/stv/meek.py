import numpy as np
from numpy.typing import NDArray
from .numpy_stv_base import NumpySTVBase, NumpyElectionDataTracker
from typing import Optional
from itertools import permutations
from math import factorial


class MeekSTV(NumpySTVBase):

    def __init__(
        self,
        profile,
        m: int = 1,
        tiebreak: Optional[str] = None,
        tolerance: float = 1e-6,
        epsilon: float = 1e-6,
        max_iterations: int = 500,
    ):
        super().__init__(profile=profile, m=m, tiebreak=tiebreak)

        self._num_cands = len(self.candidates)
        self._perm_m = self.m + 1
        self.tolerance = tolerance
        self.epsilon = epsilon
        self._max_iterations = max_iterations

        self._wt_calculator = WeightVectorCalculator(self.m, self._perm_m)

        self._run_and_store()
        helper_vecs = self._data.extras.get("helper_vecs_per_round", [])
        if helper_vecs:
            self._winner_to_cand = helper_vecs[-1]["winner_to_cand"]

    def detailed_tally_per_deg(self, round=0):
        # this was a development function that will be removed before Meek goes live
        if "helper_vecs_per_round" not in self._data.extras:
            self._run_and_store()
        record = self._data.extras["helper_vecs_per_round"][round]
        fpv_vec = record["fpv_vec"]
        winner_combination_vec = record["winner_combination_vec"]
        winners = record["winner_to_cand"]

        unique_winner_combos = np.unique(winner_combination_vec).astype(int)
        total_weight_dict = {}
        exhausted_weight_dict = {}
        for perm_idx in unique_winner_combos:
            perm = idx_to_perm(perm_idx, self._perm_m)
            translated_perm = tuple([winners[i] for i in perm])

            ballot_mask = winner_combination_vec == perm_idx
            exhausted_mask = ballot_mask & (fpv_vec < 0)

            total_weight_dict[translated_perm] = sum(self._data.wt_vec[ballot_mask])
            exhausted_weight_dict[translated_perm] = sum(
                self._data.wt_vec[exhausted_mask]
            )
        return total_weight_dict, exhausted_weight_dict

    def _initialize_helpers(self):
        num_ballots = self._data.ballot_matrix.shape[0]
        pos_vec = np.zeros(num_ballots, dtype=np.int8)
        fpv_vec = self._data.ballot_matrix[
            np.arange(self._data.ballot_matrix.shape[0]), pos_vec
        ]
        winner_comb_vec = np.zeros(num_ballots)
        bool_ballot_matrix = self._data.ballot_matrix != -127
        return pos_vec, fpv_vec, winner_comb_vec, bool_ballot_matrix

    def _update_helpers(
        self,
        pos_vec=None,
        fpv_vec=None,
        winner_comb_vec=None,
        bool_ballot_matrix=None,
        winner_to_cand=[],
        new_losers=[],
    ):
        num_ballots = self._data.ballot_matrix.shape[0]
        cand_to_winner = np.zeros(self._num_cands, dtype=np.int8) - 1
        for idx, cand in enumerate(winner_to_cand):
            cand_to_winner[cand] = idx
        if pos_vec is None:
            pos_vec = np.zeros(num_ballots, dtype=np.int8)
        if fpv_vec is None:
            fpv_vec = self._data.ballot_matrix[
                np.arange(self._data.ballot_matrix.shape[0]), pos_vec
            ]
        if winner_comb_vec is None:
            winner_comb_vec = np.zeros(num_ballots)
        if bool_ballot_matrix is None:
            bool_ballot_matrix = self._data.ballot_matrix != -127
        if len(new_losers) > 0:
            bool_ballot_matrix &= ~np.isin(self._data.ballot_matrix, new_losers)
            needs_update = np.isin(fpv_vec, new_losers)
            pos_vec[needs_update] = np.argmax(
                bool_ballot_matrix[needs_update, :], axis=1
            )
            fpv_vec[needs_update] = self._data.ballot_matrix[
                needs_update, pos_vec[needs_update]
            ]
        for _ in range(len(winner_to_cand)):
            needs_update = np.isin(fpv_vec, winner_to_cand)
            winner_comb_vec[needs_update] = update_perm_idx_vectorized(
                winner_comb_vec[needs_update],
                cand_to_winner[fpv_vec[needs_update]],
                self._perm_m,
            )
            bool_ballot_matrix[needs_update, pos_vec[needs_update]] = False
            pos_vec[needs_update] = np.argmax(
                bool_ballot_matrix[needs_update, :], axis=1
            )
            fpv_vec[needs_update] = self._data.ballot_matrix[
                needs_update, pos_vec[needs_update]
            ]
        return pos_vec, fpv_vec, winner_comb_vec, bool_ballot_matrix

    def tally_calculator(
        self, fpv_vec, winner_combination_vec, keep_factors, winner_to_cand
    ):
        unique_winner_combos = np.unique(winner_combination_vec)
        overall_tallies = np.zeros(self._num_cands)
        for perm_idx in unique_winner_combos:
            ballot_mask = winner_combination_vec == perm_idx
            non_exhausted_mask = ballot_mask & (fpv_vec >= 0)

            wt_vec_for_this_permutation = self._wt_calculator.make_wt_vec(
                perm_idx, keep_factors
            )
            weights_per_winner = (
                sum(self._data.wt_vec[ballot_mask]) * wt_vec_for_this_permutation[:-1]
            )

            leftover_weight = wt_vec_for_this_permutation[-1]
            leftover_tally = np.bincount(
                fpv_vec[non_exhausted_mask],
                weights=self._data.wt_vec[non_exhausted_mask] * leftover_weight,
                minlength=self._num_cands,
            )
            leftover_tally = leftover_tally.astype(np.float64)
            leftover_tally[winner_to_cand] += weights_per_winner[: len(winner_to_cand)]
            overall_tallies += leftover_tally
        return overall_tallies

    def calibrate_keep_factors(
        self, fpv_vec, winner_combination_vec, winner_to_cand, keep_factors
    ):
        for iteration in range(self._max_iterations):
            tallies = self.tally_calculator(
                fpv_vec, winner_combination_vec, keep_factors, winner_to_cand
            )
            active_votes = sum(tallies)
            quota = self._get_threshold(
                quota_type="droop",
                total_ballot_wt=active_votes,
                floor=False,
                epsilon=self.epsilon,
            )
            if np.all(tallies[winner_to_cand] - quota < self.tolerance):
                break
            new_keep_factors = quota / tallies[winner_to_cand]
            keep_factors *= new_keep_factors
        if np.any(keep_factors > 1.0):
            print(f"Warning! Keep factors exceeded 1.0: {keep_factors}.")
        return tallies, keep_factors, iteration + 1, quota

    def meek_stv_engine(
        self,
        pos_vec,
        fpv_vec,
        winner_combination_vec,
        bool_ballot_matrix,
        winner_to_cand,
        hopeful,
        starting_keep_factors,
        tiebreak_record,
    ):
        # 1) calibrate keep factors
        # 2) record info and determine loser/winner (non-simultaneous is same as simultaneous, and cleaner)
        # 3) update helpers
        keep_factors = starting_keep_factors.copy()
        tallies, keep_factors, iterations, current_quota = self.calibrate_keep_factors(
            fpv_vec, winner_combination_vec, winner_to_cand, keep_factors=keep_factors
        )
        non_winner_tallies = np.copy(tallies)
        non_winner_tallies[winner_to_cand] = -1
        if np.any(non_winner_tallies > current_quota):  # elect highest winner
            max_tally = np.max(non_winner_tallies)
            highest_cands = np.where(non_winner_tallies == max_tally)[0].tolist()
            if len(highest_cands) > 1:
                winner, tiebreak_record = self._run_winner_tiebreak(
                    tied_winners=highest_cands,
                    turn=0,
                    mutant_tiebreak_record=tiebreak_record,
                )
            else:
                winner = int(highest_cands[0])
                tiebreak_record.append({})
            winner_to_cand.append(winner)
            keep_factors = np.append(keep_factors, np.float64(1.0))
            round_type = "election"
            new_losers = []
        else:  # eliminate loser
            hopeful_tallies = tallies[hopeful]
            lowest_tally = np.min(hopeful_tallies)
            lowest_cands = np.where(hopeful_tallies == lowest_tally)[0].tolist()
            if len(lowest_cands) > 1:
                print(
                    f"Tie for lowest hopeful tallies among candidates {[hopeful[i] for i in lowest_cands]} with tally {lowest_tally}. Running tiebreak."
                )
                new_loser, tiebreak_record = self._run_loser_tiebreak(
                    tied_losers=[hopeful[i] for i in lowest_cands],
                    turn=0,
                    mutant_tiebreak_record=tiebreak_record,
                )
            else:
                new_loser = int(hopeful[lowest_cands[0]])
                tiebreak_record.append({})
            new_losers = [new_loser]
            if new_loser not in hopeful:
                raise ValueError(
                    f"Tried to eliminate candidate {new_loser} who is not in hopeful list {hopeful}, tallies {tallies}, hopeful_tallies {hopeful_tallies}."
                )
            hopeful.remove(new_loser)
            round_type = "elimination"
            winner = None
        pos_vec, fpv_vec, winner_combination_vec, bool_ballot_matrix = (
            self._update_helpers(
                pos_vec=pos_vec,
                fpv_vec=fpv_vec,
                winner_comb_vec=winner_combination_vec,
                bool_ballot_matrix=bool_ballot_matrix,
                winner_to_cand=winner_to_cand,
                new_losers=new_losers,
            )
        )
        return (
            tallies,
            keep_factors,
            iterations,
            current_quota,
            pos_vec,
            fpv_vec,
            winner_combination_vec,
            bool_ballot_matrix,
            round_type,
            new_losers,
            [winner],
            winner_to_cand,
            hopeful,
            tiebreak_record,
        )

    def _run_election(self, data: NumpyElectionDataTracker):
        fpv_by_round = []
        quota_by_round = []
        helper_vecs_per_round = []
        play_by_play = []
        tiebreak_record = []

        pos_vec, fpv_vec, winner_combination_vec, bool_ballot_matrix = (
            self._initialize_helpers()
        )
        winner_to_cand = []
        _, keep_factors, _, _ = self.calibrate_keep_factors(
            fpv_vec=fpv_vec,
            winner_combination_vec=winner_combination_vec,
            winner_to_cand=winner_to_cand,
            keep_factors=np.ones(len(winner_to_cand)),
        )

        hopeful = np.arange(0, self._num_cands).tolist()

        round_number = 0
        while len(winner_to_cand) < self.m:
            (
                tallies,
                keep_factors,
                iterations,
                current_quota,
                pos_vec,
                fpv_vec,
                winner_combination_vec,
                bool_ballot_matrix,
                round_type,
                new_losers,
                new_winners,
                winner_to_cand,
                hopeful,
                tiebreak_record,
            ) = self.meek_stv_engine(
                pos_vec,
                fpv_vec,
                winner_combination_vec,
                bool_ballot_matrix,
                winner_to_cand,
                hopeful,
                keep_factors,
                tiebreak_record,
            )
            fpv_by_round.append(tallies.copy())
            quota_by_round.append(current_quota)
            # tiebreak_record.append #TODO: tiebreaks
            helper_vecs_per_round.append(
                {
                    "fpv_vec": fpv_vec.copy(),
                    "winner_combination_vec": winner_combination_vec.copy(),
                    "winner_to_cand": winner_to_cand.copy(),
                    "quota": current_quota,
                }
            )
            play = {
                "round_number": int(round_number),
                "keep_factors": keep_factors.copy(),
                # "quota": current_quota,
                "iterations": iterations,
                "round_type": round_type,
            }
            if play["round_type"] == "elimination":
                play["loser"] = new_losers
            elif play["round_type"] == "election":
                play["winners"] = new_winners
            play_by_play.append(play)
            round_number += 1
        final_tallies, keep_factors, _, final_quota = self.calibrate_keep_factors(
            fpv_vec=fpv_vec,
            winner_combination_vec=winner_combination_vec,
            winner_to_cand=winner_to_cand,
            keep_factors=keep_factors,
        )
        # print(keep_factors)
        fpv_by_round.append(final_tallies)
        quota_by_round.append(final_quota)

        data.extras["helper_vecs_per_round"] = helper_vecs_per_round
        data.extras["quota_by_round"] = quota_by_round
        data.extras["winner_to_cand"] = winner_to_cand

        return fpv_by_round, play_by_play, tiebreak_record


class WeightVectorCalculator:
    """
    Lightweight calculator that knows how ballot weight is split across a
    permutation of winners for a given keep-factor vector.
    """

    def __init__(self, num_winners, m=None):
        """
        Args:
            num_winners: number of possible winners (L)
            m: parameter for idx_to_perm; should be L + 1 (elements 0..L-1)
        """
        self.num_winners = num_winners
        self.m = m if m is not None else num_winners + 1
        if self.m != self.num_winners + 1:
            raise ValueError(
                f"WeightVectorCalculator expects m = num_winners + 1 (got {self.m} vs {self.num_winners})."
            )

        # number of permutations of up to L elements: sum_{l=0..L} L!/(L-l)!
        self.num_perms = sum(
            factorial(num_winners) // factorial(num_winners - l)
            for l in range(num_winners + 1)
        )

        # Precompute the canonical permutations so make_wt_vec can stay simple.
        self.perms = [
            idx_to_perm(perm_idx, self.m) for perm_idx in range(self.num_perms)
        ]

    def make_wt_vec(self, perm_idx, keep_factors):
        """
        Fast weight vector calculation using precomputed structure.

        Args:
            perm_idx: Index of the permutation
            keep_factors: Array of keep factors for each winner

        Returns:
            Weight vector of length num_winners + 1
        """
        perm = self.perms[int(perm_idx)]
        wt_vec = np.zeros(self.num_winners + 1)

        carry = 1.0
        for winner_idx in perm:
            if winner_idx >= len(keep_factors):
                raise IndexError(
                    f"Permutation index {perm_idx} refers to winner {winner_idx}, "
                    f"but keep_factors has length {len(keep_factors)}."
                )
            wt_vec[winner_idx] = carry * keep_factors[winner_idx]
            carry *= 1 - keep_factors[winner_idx]

        wt_vec[-1] = carry
        return wt_vec

    def make_wt_vec_vectorized(self, perm_indices, keep_factors):
        """
        Vectorized version that computes weight vectors for multiple permutations.

        Args:
            perm_indices: Array of permutation indices
            keep_factors: Array of keep factors for each winner

        Returns:
            2D array where each row is a weight vector
        """
        n = len(perm_indices)
        result = np.zeros((n, self.num_winners + 1))

        for i, perm_idx in enumerate(perm_indices):
            result[i] = self.make_wt_vec(perm_idx, keep_factors)

        return result


def update_perm_idx_vectorized(idx_vec, j_vec, m):
    """
    Vectorized version of update_perm_idx.

    Returns indices that result from appending elements j_vec to permutations at idx_vec.

    Args:
        idx_vec: Array of indices of original permutations
        j_vec: Array of elements to append (0 <= j < m-1)
        m: The parameter defining the range; pass num_winners + 1

    Returns:
        Array of indices of extended permutations

    Example:
        update_perm_idx_vectorized(np.array([2, 1]), np.array([0, 1]), 3)
        returns np.array([4, 3])
    """
    idx_vec = np.asarray(idx_vec, dtype=int)
    j_vec = np.asarray(j_vec, dtype=int)

    elements = m - 1
    n = len(idx_vec)
    result = np.zeros(n, dtype=int)

    # Precompute factorial values
    fact = np.array([factorial(i) for i in range(elements + 1)])

    # For each index, determine its length group
    lengths = np.zeros(n, dtype=int)
    pos_in_length = np.zeros(n, dtype=int)

    for i in range(n):
        current_idx = 0
        for l in range(m):
            count = fact[elements] // fact[elements - l] if l <= elements else 0
            if current_idx + count > idx_vec[i]:
                lengths[i] = l
                pos_in_length[i] = idx_vec[i] - current_idx
                break
            current_idx += count

    # Process each element
    for i in range(n):
        length = lengths[i]
        new_length = length + 1

        # Count of permutations before the new length group
        idx_before_new_length = 0
        for l in range(new_length):
            count = fact[elements] // fact[elements - l] if l <= elements else 0
            idx_before_new_length += count

        if length == 0:
            # Original is empty (), extended is (j,)
            result[i] = idx_before_new_length + j_vec[i]
        else:
            # Decode the original permutation
            available = list(range(elements))
            perm_list = []
            remaining_pos = pos_in_length[i]

            for pos in range(length):
                remaining_slots = length - pos - 1
                perms_per_choice = fact[elements - pos - 1] // fact[elements - length]

                choice_idx = remaining_pos // perms_per_choice
                perm_list.append(available[choice_idx])
                available.pop(choice_idx)
                remaining_pos %= perms_per_choice

            # Check if j is already in the permutation
            if j_vec[i] in perm_list:
                raise ValueError(
                    f"Element {j_vec[i]} is already in the permutation at index {i}"
                )

            # Create the extended permutation
            extended_perm = perm_list + [j_vec[i]]

            # Find the position of this extended permutation in the new length group
            available_new = list(range(elements))
            new_pos = 0

            for pos in range(new_length):
                elem = extended_perm[pos]
                elem_idx = available_new.index(elem)

                remaining_slots = new_length - pos - 1
                perms_per_choice = (
                    fact[elements - pos - 1] // fact[elements - new_length]
                )
                new_pos += elem_idx * perms_per_choice

                available_new.remove(elem)

            result[i] = idx_before_new_length + new_pos

    return result


def idx_to_perm(idx, m):
    """
    Maps an index to a canonical permutation of elements 0 to m-2.
    (In this file, m is normally num_winners + 1.)

    Canonical order:
    - First by length (0, 1, 2, ..., m-1)
    - Within each length, lexicographically

    Args:
        idx: The index of the permutation
        m: The parameter defining the range (elements are 0 to m-2)

    Returns:
        A tuple representing the permutation

    Example for m=3:
        idx_to_perm(0, 3) = ()
        idx_to_perm(1, 3) = (0,)
        idx_to_perm(2, 3) = (1,)
        idx_to_perm(3, 3) = (0, 1)
        idx_to_perm(4, 3) = (1, 0)
    """
    elements = list(range(m - 1))
    current_idx = 0

    # Generate permutations by length
    for length in range(m):
        # Get all permutations of this length
        perms = sorted(permutations(elements, length))

        # Check if our target index is in this length group
        if current_idx + len(perms) > idx:
            # Found the right length group
            perm_idx = idx - current_idx
            return perms[perm_idx]

        current_idx += len(perms)

    raise IndexError(f"Index {idx} out of range for m={m}")


def perm_to_idx(perm, m):
    """
    Maps a permutation to its canonical index.

    Args:
        perm: A tuple representing the permutation
        m: The parameter defining the range (elements are 0 to m-2)

    Returns:
        The index of the permutation
    """
    elements = list(range(m - 1))
    current_idx = 0
    length = len(perm)

    # Add counts for all shorter lengths
    for l in range(length):
        perms = list(permutations(elements, l))
        current_idx += len(perms)

    # Find position within permutations of this length
    perms_of_length = sorted(permutations(elements, length))
    current_idx += perms_of_length.index(perm)

    return current_idx
