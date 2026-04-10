from dataclasses import dataclass
from functools import lru_cache
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import DTypeLike, NDArray

from votekit import RankProfile
from votekit.elections.election_types.ranking.stv.numpy_stv_base import (
    ElectionPlay,
    NumpyElectionDataTracker,
    NumpySTVBase,
    TiebreakType,
)


@dataclass(frozen=True, slots=True)
class KeepFactorCalibrationCache:
    """
    Round-local compressed representation of the data needed to calibrate keep factors.

    Stores the compressed support of the current winner_combination_vec,
        together with the ballot-to-support mapping and the grouped masses
        needed for keep-factor calibration and final tally reconstruction.
    N is the number of ballots in the ballot matrix.
    P is the number of unique winner combinations observed in the current round.
    Lmax is the maximum ranking length of the profile.
        In the future, Lmax could also refer to the maximum number of winners a
        ballot is allowed to transfer to.

    Attributes:
        support_comb_ids: the unique winner_combination_vec values observed
            in this round, in ascending order.
        ballot_to_support_row: maps each index of the ballot matrix to
            the index of its winner combination in support_comb_ids.
        sliced_winner_matrix: view of the winner_permutation_matrix containing
            only the observed winner combinations.
        valid_mask: boolean mask indicating which entries of sliced_winner_matrix
            contain valid winner positions (as opposed to padding).
        permutation_lengths: the number of winners in each
            observed winner combination.
        support_total_mass: The total starting ballot weight attached to
            each observed winner-combination.
        support_nonexhausted_mass: The total starting ballot weight attached to
            nonexhausting ballots in each observed winner-combination.
        exhausted_mask: Ballot-level boolean mask for whether each
            ballot row is currently exhausted.
        fpv_vec: The current first-preference candidate for each ballot row.
        initial_wt_vec: The initial weight of each ballot row.
    """

    support_comb_ids: NDArray  # shape (P,)
    ballot_to_support_row: NDArray  # shape (N,)
    sliced_winner_matrix: NDArray  # shape (P, Lmax), padded with safe values
    valid_mask: NDArray  # shape (P, Lmax), True where sliced_winner_matrix is meaningful
    permutation_lengths: NDArray  # shape (P,)
    support_total_mass: NDArray  # shape (P,)
    support_nonexhausted_mass: NDArray  # shape (P,)
    exhausted_mask: NDArray  # shape (N,)
    fpv_vec: NDArray  # shape (N,)
    initial_wt_vec: NDArray  # shape (N,)


MutableLoserBundle: TypeAlias = tuple[
    list[int],
    list[dict[frozenset[str], tuple[frozenset[str], ...]]],
    NDArray,
    NDArray,
    NDArray,
]


class MeekSTV(NumpySTVBase):
    """
    STV variant with keep factors instead of transfer values.

    Winners can still receive transfers after being seated,
        in which case they decrease their keep factor.
    Quota is recalculated each round, and keep factors are lowered
        until all seated winner tallies are within tolerance of quota.
    Election rounds are simultaneous by definition.
    Loser tiebreaks use first round tallies by default, but can be specified otherwise.
    """

    def __init__(
        self,
        profile: RankProfile,
        n_seats: int = 1,
        tiebreak: TiebreakType | None = None,
        tolerance: float | None = 1e-6,
        epsilon: float | None = 1e-6,
        max_iterations: int | None = 500,
    ):
        """
        Initialize a Meek STV election with advanced options.

        Args:
            profile (RankProfile): RankProfile to run election on.
            n_seats (int): Number of seats to be elected. Defaults to 1.
            tiebreak (TiebreakType | None, optional): Method to be used if a tiebreak is needed. Accepts
                "borda", "random", and "cambridge_random". Defaults to None, in which case a ValueError is
                raised if a tiebreak is needed.
            tolerance (float, optional): Margin by which a winner's tally is allowed to exceed quota
                without further keep factor adjustments. Defaults to 1e-6.
            epsilon (float, optional): Small value added to quota to ensure that no more than n_seats candidates
                can have a quota's worth of the active votes within a round. Defaults to 1e-6.
            max_iterations (int, optional): Maximum number of keep factor iterations to go through
                before breaking out of the calibration process. Defaults to 500.
        """
        super().__init__(profile=profile, n_seats=n_seats, tiebreak=tiebreak)

        if n_seats > 10:
            raise NotImplementedError(
                "Meek STV with more than 10 seats is not currently supported due to the combinatorial explosion of winner combinations."
            )
            self._dense = False
        else:
            self._dense = True  # TODO: implement a backup protocol when n_seats, L are large
        self._num_cands = len(self.candidates)
        self._max_ranking_length = min(profile.max_ranking_length, self._num_cands)
        self.tolerance = 1e-6 if tolerance is None else float(tolerance)
        self.epsilon = 1e-6 if epsilon is None else float(epsilon)
        self._max_iterations = 500 if max_iterations is None else int(max_iterations)

        self._run_and_store()

    def _run_election(
        self, mutable_data_tracker: NumpyElectionDataTracker
    ) -> NumpyElectionDataTracker:
        """
        Core election logic for Meek STV.

        Args:
            mutable_data_tracker (NumpyElectionDataTracker): The initialized data tracker with
                the profile converted to numpy arrays.

        Returns:
            mutable_data_tracker (NumpyElectionDataTracker): The updated data tracker with
                election results.
        """
        ballot_matrix = mutable_data_tracker.ballot_matrix
        initial_wt_vec = np.copy(mutable_data_tracker.wt_vec)
        winner_combination_vec = np.zeros_like(initial_wt_vec, dtype=np.int64)
        winner_bitstring_vec = np.zeros_like(initial_wt_vec, dtype=np.int32)
        pos_vec = np.zeros_like(initial_wt_vec, dtype=np.int8)
        winner_list = []
        fpv_vec = np.copy(ballot_matrix[:, 0])
        n_seats = self.n_seats

        fpv_scores_by_round = []
        num_iterations_by_round = []
        keep_factor_by_round = []
        play_by_play: list[ElectionPlay] = []
        round_number = 0
        eliminated_candidates: list[int] = []
        tiebreak_record: list[dict[frozenset[str], tuple[frozenset[str], ...]]] = []
        bool_ballot_matrix: NDArray = np.ones_like(ballot_matrix, dtype=bool)

        if self._dense:
            winner_combination_matrix = _permutation_matrix_constructor(
                n_seats, self._max_ranking_length, dtype=np.dtype(np.int8)
            )

        winner_combination_mutant_bundle = (
            winner_combination_vec,
            winner_bitstring_vec,
            fpv_vec,
            bool_ballot_matrix,
            pos_vec,
        )
        keep_factor_calibrator_bundle = (
            initial_wt_vec,
            fpv_vec,
            winner_list,
            winner_combination_vec,
            winner_combination_matrix,
        )
        loser_mutant_bundle: MutableLoserBundle = (
            eliminated_candidates,
            tiebreak_record,
            fpv_vec,
            pos_vec,
            bool_ballot_matrix,
        )

        while len(winner_list) < n_seats:
            tallies, keep_factors, current_quota, num_iterations = self._keep_factor_calibrator(
                *keep_factor_calibrator_bundle
            )
            fpv_scores_by_round.append(tallies.copy())
            keep_factor_by_round.append(keep_factors.copy())
            num_iterations_by_round.append(num_iterations)
            masked_tallies = np.where(
                np.isin(np.arange(len(tallies)), winner_list),
                0,
                tallies,
            )
            while np.any(masked_tallies >= current_quota):
                winners = np.where(tallies >= current_quota)[0]
                winners = winners[~np.isin(winners, winner_list)]
                winners = winners[np.argsort(-tallies[winners])]
                winner_list.extend(winners)
                tiebreak_record.append({})
                winner_combination_mutant_bundle = self._update_winner_comb_vec(
                    *winner_combination_mutant_bundle, all_winners=winner_list
                )
                play_by_play.append(
                    ElectionPlay(
                        round_number=int(round_number),
                        winners=[int(c) for c in winners],
                        threshold=current_quota,
                        round_type="election",
                    )
                )
                round_number += 1
                tallies, keep_factors, current_quota, num_iterations = self._keep_factor_calibrator(
                    *keep_factor_calibrator_bundle
                )
                keep_factor_by_round.append(keep_factors.copy())
                num_iterations_by_round.append(num_iterations)
                fpv_scores_by_round.append(tallies.copy())
                masked_tallies = np.where(
                    np.isin(np.arange(len(tallies)), winner_list),
                    0,
                    tallies,
                )
            if len(winner_list) == n_seats:
                break
            loser_idx, loser_mutant_bundle = self._find_and_eliminate_loser(
                tallies, round_number, *loser_mutant_bundle
            )
            winner_combination_mutant_bundle = self._update_winner_comb_vec(
                *winner_combination_mutant_bundle, all_winners=winner_list
            )
            play_by_play.append(
                ElectionPlay(
                    round_number=int(round_number),
                    loser=[int(loser_idx)],
                    threshold=current_quota,
                    round_type="elimination",
                )
            )
            round_number += 1
        mutable_data_tracker.fpv_by_round = fpv_scores_by_round
        mutable_data_tracker.play_by_play = play_by_play
        mutable_data_tracker.tiebreak_record = tiebreak_record
        mutable_data_tracker.extras["keep_factor_by_round"] = keep_factor_by_round
        mutable_data_tracker.extras["num_iterations_by_round"] = num_iterations_by_round
        return mutable_data_tracker

    def _keep_factor_calibrator(
        self,
        initial_wt_vec: NDArray,
        fpv_vec: NDArray,
        winners: list[int],
        winner_combination_vec: NDArray,
        winner_combination_matrix: NDArray,
    ) -> tuple[NDArray, NDArray, float, int]:
        """
        Runs steps 1-3 of the keep factor calibration process.

        Step 1: condense and cache the information about the observed winner combinations.
        Step 2: iteratively adjust keep factors and re-compute the tallies of *winners only*
            until all winners are within tolerance of quota.
        Step 3: use the leftover mass of ballots to compute the tallies of non-winner candidates.
        If there are no elected winners yet, skip steps 1 and 2.

        Args:
            initial_wt_vec: (NDArray) initial weight (i.e. multiplicity) of each
                ballot in the ballot_matrix.
            fpv_vec: (NDArray) current first preference candidate of each
                ballot in the ballot_matrix.
            winners: list of currently seated winners, indexed according to their
                position in self.candidates.
            winner_combination_vec: (NDArray) current winner combination index for each ballot.
            winner_combination_matrix: (NDArray) matrix mapping winner combination indices
                to winner combinations; each row is a different combination.

        Returns:
            tallies: (NDArray) final tallies for all candidates after keep factors have converged.
            keep_factors: (NDArray) final keep factors for all seated winners.
            quota: the quota calculated in the final iteration of the calibration process.
            iterations_used: the number of iterations of keep factor adjustment that were used.
        """

        if len(winners) == 0:
            exhausting_mask = fpv_vec < 0
            active_votes = float(initial_wt_vec[~exhausting_mask].sum())
            quota = self._get_threshold(
                "droop",
                active_votes,
                floor=False,
                epsilon=self.epsilon,
            )
            keep_factors = np.array([], dtype=np.float64)
            tallies = np.bincount(
                fpv_vec[~exhausting_mask],
                weights=initial_wt_vec[~exhausting_mask],
                minlength=self._num_cands,
            )
            iterations = 0

        else:
            cache = self._build_keep_factor_calibration_cache(
                initial_wt_vec=initial_wt_vec,
                fpv_vec=fpv_vec,
                winner_combination_vec=winner_combination_vec,
                winner_combination_matrix=winner_combination_matrix,
            )

            keep_factors, winner_tallies, leftover_factor, quota, iterations = (
                self._iterate_keep_factors_from_cache(
                    cache=cache,
                    num_winners=len(winners),
                )
            )

            tallies = self._finalize_tallies_from_cache(
                cache=cache,
                winners=winners,
                winner_tallies=winner_tallies,
                leftover_factor_by_support=leftover_factor,
            )

        return tallies, keep_factors, quota, iterations

    def _build_keep_factor_calibration_cache(
        self,
        initial_wt_vec: NDArray,
        fpv_vec: NDArray,
        winner_combination_vec: NDArray,
        winner_combination_matrix: NDArray,
    ) -> KeepFactorCalibrationCache:
        """
        Compress and cache ballot-level data about winner combinations.

        Args:
            initial_wt_vec: (NDArray) initial weight (i.e. multiplicity) of each
                ballot in the ballot_matrix.
            fpv_vec: (NDArray) current first preference candidate of each
                ballot in the ballot_matrix.
            winner_combination_vec: (NDArray) current winner combination index for each ballot.
            winner_combination_matrix: (NDArray) matrix mapping winner combination indices
                to winner combinations; each row is a different combination.
                In sparse mode, we might use a list of arrays instead of this matrix,
                and materialize it as an array at this stage of the process.

        Returns:
            KeepFactorCalibrationCache: a cache ready to be decoded by the later stages of
                the calibration process.
        """
        support_comb_ids, ballot_to_support_row = np.unique(
            winner_combination_vec,
            return_inverse=True,
        )

        exhausted_mask = fpv_vec < 0

        support_total_mass = np.bincount(
            ballot_to_support_row,
            weights=initial_wt_vec,
            minlength=len(support_comb_ids),
        ).astype(np.int32)

        support_nonexhausted_mass = np.bincount(
            ballot_to_support_row[~exhausted_mask],
            weights=initial_wt_vec[~exhausted_mask],
            minlength=len(support_comb_ids),
        ).astype(np.int32)

        raw_view_of_permutation_matrix = winner_combination_matrix[support_comb_ids]

        valid_mask = raw_view_of_permutation_matrix >= 0
        permutation_lengths = valid_mask.sum(axis=1).astype(np.int8)

        sliced_winner_matrix = raw_view_of_permutation_matrix.copy()
        sliced_winner_matrix[~valid_mask] = 0

        return KeepFactorCalibrationCache(
            support_comb_ids=support_comb_ids,
            ballot_to_support_row=ballot_to_support_row,
            sliced_winner_matrix=sliced_winner_matrix,
            valid_mask=valid_mask,
            permutation_lengths=permutation_lengths,
            support_total_mass=support_total_mass,
            support_nonexhausted_mass=support_nonexhausted_mass,
            exhausted_mask=exhausted_mask,
            fpv_vec=fpv_vec,
            initial_wt_vec=initial_wt_vec,
        )

    def _iterate_keep_factors_from_cache(
        self,
        cache: KeepFactorCalibrationCache,
        num_winners: int,  # should I have an optional argument here to store each iteration somewhere?
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        """
        Logic for the kernel of the keep factor calibration process.

        Deliberately initializes keep factors to 1 (rather than start from the previous round's
            keep factors) for auditability.

        Args:
            cache: the compressed data cache for the current round, containing information
                about the observed winner combinations and their masses.
            num_winners: the number of winners currently seated, which determines
                the shape of keep_factors and winner_tallies.

        Returns:
            keep_factors: (NDarray) final keep factors for all seated winners after convergence.
            winner_tallies: (NDarray) final tallies for all seated winners after convergence.
            leftover_factor_by_support: (NDarray) fraction of each support row entry's mass that
                remains after transferring through their winners.
            quota: (float) the final value of quota when the process converged.
            iterations_used: (int) the number of iterations that ran before convergence.
        """
        keep_factors = np.ones(num_winners, dtype=np.float64)

        P, Lmax = cache.sliced_winner_matrix.shape

        k_mat = np.zeros((P, Lmax), dtype=np.float64)
        prefix = np.ones((P, Lmax), dtype=np.float64)
        carry_before = np.zeros((P, Lmax), dtype=np.float64)
        contrib = np.zeros((P, Lmax), dtype=np.float64)

        winner_tallies = np.zeros(num_winners, dtype=np.float64)
        leftover_factor = np.ones(P, dtype=np.float64)

        nonempty_rows = cache.permutation_lengths > 0
        nonempty_idx = np.where(nonempty_rows)[0]

        iterations_used = 0

        for iteration in range(self._max_iterations):  # let me know if you want these comments gone
            iterations_used = iteration + 1

            # k_mat[p, j] = keep factor for the winner at position j of support row p
            k_mat.fill(0.0)
            k_mat[cache.valid_mask] = keep_factors[cache.sliced_winner_matrix[cache.valid_mask]]

            # prefix = inclusive prefix products of (1 - k) along each support row
            prefix.fill(1.0)
            prefix[cache.valid_mask] -= k_mat[cache.valid_mask]
            np.cumprod(prefix, axis=1, out=prefix)

            # carry_before = exclusive prefix products
            carry_before.fill(0.0)
            carry_before[:, 0] = 1.0
            if Lmax > 1:
                carry_before[:, 1:] = prefix[:, :-1]
            carry_before[~cache.valid_mask] = 0.0

            # contrib[p, j] = total mass in support row p assigned to winner at position j
            contrib[:] = carry_before
            contrib *= k_mat
            contrib *= cache.support_total_mass[:, None]
            contrib[~cache.valid_mask] = 0.0

            # Scatter-add support-row/position contributions into winner tallies.
            winner_tallies.fill(0.0)
            for j in range(Lmax):
                not_empty_position = cache.valid_mask[:, j]
                if np.any(not_empty_position):
                    np.add.at(
                        winner_tallies,
                        cache.sliced_winner_matrix[not_empty_position, j],
                        contrib[not_empty_position, j],
                    )

            # leftover_factor[p] = fraction of a ballot in support row p
            # that remains after transferring through their winners.
            leftover_factor.fill(1.0)
            if len(nonempty_idx) > 0:
                leftover_factor[nonempty_idx] = prefix[
                    nonempty_idx,
                    cache.permutation_lengths[nonempty_idx] - 1,
                ]

            # Active votes = winner tallies + nonexhausted leftover mass.
            active_votes = float(
                winner_tallies.sum() + np.dot(cache.support_nonexhausted_mass, leftover_factor)
            )

            quota = self._get_threshold(
                "droop",
                active_votes,
                floor=False,
                epsilon=self.epsilon,
            )

            if np.all(winner_tallies <= quota + self.tolerance):
                break

            # Meek update: k_i <- k_i * min(q / T_i, 1)
            scale = np.ones_like(keep_factors)
            positive = winner_tallies > 0.0
            scale[positive] = np.minimum(quota / winner_tallies[positive], 1.0)
            keep_factors *= scale

        return keep_factors, winner_tallies, leftover_factor, quota, iterations_used

    def _finalize_tallies_from_cache(
        self,
        cache: KeepFactorCalibrationCache,
        winners: list[int],
        winner_tallies: NDArray,
        leftover_factor_by_support: NDArray,
    ) -> NDArray:
        """
        After keep factors have converged, compute full candidate tallies for other candidates.

        Args:
            cache (KeepFactorCalibrationCache): the compressed data cache for the current round.
            winners (list[int]): the list of seated winners in the current round.
            winner_tallies (NDArray): the already-computed tallies of the winning candidates.
            leftover_factor_by_support (NDArray): the fractional ballot weight
                (not including multiplicity) for each observed winner combination.

        Returns:
            tallies (NDArray): the final tallies for all candidates.
        """
        actual_wt_vec = (
            cache.initial_wt_vec * leftover_factor_by_support[cache.ballot_to_support_row]
        )

        tallies = np.bincount(
            cache.fpv_vec[~cache.exhausted_mask],
            weights=actual_wt_vec[~cache.exhausted_mask],
            minlength=self._num_cands,
        ).astype(np.float64)

        if len(winners) > 0:
            tallies[np.asarray(winners, dtype=int)] = winner_tallies

        return tallies

    def _update_winner_comb_vec(
        self,
        mutant_winner_comb_vec: NDArray,
        mutant_winner_bitstring_vec: NDArray,
        mutant_fpv_vec: NDArray,
        mutant_bool_ballot_matrix: NDArray,
        mutant_pos_vec: NDArray,
        all_winners: list[int],
        n_seats: int | None = None,
        L: int | None = None,
    ):
        """
        Advance ballots whose current first-preference candidate is already seated.

        Updates both the winner-combination index and the winner bitstring until
        no ballot currently points to a seated winner.

        Args:
            mutant_winner_comb_vec: (NDArray) the previous winner combination index
                for each ballot.
            mutant_winner_bitstring_vec: (NDArray) the previous winner bitstring
                for each ballot.
            mutant_fpv_vec: (NDArray) the current first-preference vector for each ballot.
            mutant_bool_ballot_matrix: (NDArray) the boolean ballot matrix indicating
                available positions for each ballot.
            mutant_pos_vec: (NDArray) the current position of the fpv on each ballot.
            all_winners: (list[int]) the list of seated winners in the current round,
                in the order they were seated in.
            n_seats: (int | None) the number of seats to be filled.
                Defaults to self.n_seats.
            L: (int | None) the maximum ranking length.
                Defaults to self._max_ranking_length.

        Returns:
            mutant_winner_comb_vec: (NDArray) mutated in place.
            mutant_winner_bitstring_vec: (NDArray) mutated in place.
            mutant_fpv_vec: (NDArray) mutated in place.
            mutant_bool_ballot_matrix: (NDArray) mutated in place.
            mutant_pos_vec: (NDArray) mutated in place.

        """
        if n_seats is None:
            n_seats = self.n_seats
        if L is None:
            L = self._max_ranking_length

        if len(all_winners) == 0:
            return (
                mutant_winner_comb_vec,
                mutant_winner_bitstring_vec,
                mutant_fpv_vec,
                mutant_bool_ballot_matrix,
                mutant_pos_vec,
            )

        winner_array = np.asarray(all_winners, dtype=int)

        cand_to_winner_pos = np.full(self._num_cands, -1, dtype=np.int32)
        cand_to_winner_pos[winner_array] = np.arange(winner_array.size, dtype=np.int32)

        max_passes = winner_array.size
        num_passes = 0

        while True:
            current_winner_pos = np.full(mutant_fpv_vec.shape, -1, dtype=np.int32)
            active_rows = mutant_fpv_vec >= 0
            current_winner_pos[active_rows] = cand_to_winner_pos[mutant_fpv_vec[active_rows]]

            needs_update = current_winner_pos >= 0
            if not np.any(needs_update):
                break

            num_passes += 1
            if num_passes > max_passes:
                raise RuntimeError(
                    "Winner-combination update exceeded the expected number of passes. "
                    "This suggests some ballots are repeatedly landing on seated winners "
                    "without making progress."
                )

            updated_comb, updated_bits = _vectorized_perm_updater(
                mutant_winner_comb_vec[needs_update],
                self._num_cands,
                L,
                mutant_winner_bitstring_vec[needs_update],
                current_winner_pos[needs_update],
            )
            mutant_winner_comb_vec[needs_update] = updated_comb
            mutant_winner_bitstring_vec[needs_update] = updated_bits

            mutant_bool_ballot_matrix[needs_update, mutant_pos_vec[needs_update]] = False

            mutant_pos_vec[needs_update] = mutant_bool_ballot_matrix[needs_update].argmax(axis=1)

            mutant_fpv_vec[needs_update] = self._data.ballot_matrix[
                needs_update,
                mutant_pos_vec[needs_update],
            ]

        return (
            mutant_winner_comb_vec,
            mutant_winner_bitstring_vec,
            mutant_fpv_vec,
            mutant_bool_ballot_matrix,
            mutant_pos_vec,
        )

    def _find_and_eliminate_loser(
        self,
        tallies: NDArray,
        round_number: int,
        mutant_eliminated_candidates: list[int],
        mutant_tiebreak_record: list[dict[frozenset[str], tuple[frozenset[str], ...]]],
        mutant_fpv_vec: NDArray,
        mutant_pos_vec: NDArray,
        mutant_bool_ballot_matrix: NDArray,
    ) -> tuple[int, MutableLoserBundle]:
        """
        Identifies the candidate with the lowest tally and eliminates them in place.

        Breaks and records ties as needed.
        Args:
            tallies: (NDArray) the current tallies for all candidates.
            round_number: (int) the current round number, used for tiebreak record keeping.
            mutant_eliminated_candidates: (list[int]) list of already eliminated candidates.
            mutant_tiebreak_record: (list[dict[frozenset[str], tuple[frozenset[str], ...]]])
                A record of all tiebreaks containing one entry per round.
            mutant_fpv_vec: (NDArray) the current first-preference vector for each ballot.
            mutant_pos_vec: (NDArray) the current position of the fpv on each ballot.
            mutant_bool_ballot_matrix: (NDArray) the boolean ballot matrix indicating
                available positions for each ballot.

        Returns:
            loser_idx: (int) the index of the candidate to be eliminated.
            mutant_eliminated_candidates: (list[int]) mutated in place.
            mutant_tiebreak_record: (list[dict[frozenset[str], tuple[frozenset[str], ...]]])
                mutated in place.
            mutant_fpv_vec: (NDArray) mutated in place.
            mutant_pos_vec: (NDArray) mutated in place.
            mutant_bool_ballot_matrix: (NDArray) mutated in place.
        """
        masked_tallies: NDArray = np.where(
            np.isin(np.arange(len(tallies)), mutant_eliminated_candidates),
            np.inf,
            tallies,
        )
        if np.count_nonzero(masked_tallies == np.min(masked_tallies)) > 1:
            potential_losers: list[int] = (
                np.where(masked_tallies == masked_tallies.min())[0].astype(int).tolist()
            )
            loser_idx, mutant_tiebreak_record = self._run_loser_tiebreak(
                potential_losers, round_number, mutant_tiebreak_record
            )
        else:
            loser_idx = int(np.argmin(masked_tallies))
            mutant_tiebreak_record.append({})
        mutant_eliminated_candidates.append(loser_idx)
        mutant_bool_ballot_matrix &= ~np.isin(self._data.ballot_matrix, loser_idx)  # same
        rows_with_loser_fpv = mutant_fpv_vec == loser_idx
        allowed_pos_matrix = mutant_bool_ballot_matrix[rows_with_loser_fpv]
        mutant_pos_vec[rows_with_loser_fpv] = allowed_pos_matrix.argmax(axis=1)
        mutant_fpv_vec[rows_with_loser_fpv] = (
            self._data.ballot_matrix[  # this would be self._data.ballot_matrix in the class context
                rows_with_loser_fpv, mutant_pos_vec[rows_with_loser_fpv]
            ]
        )

        return (
            loser_idx,
            (
                mutant_eliminated_candidates,
                mutant_tiebreak_record,
                mutant_fpv_vec,
                mutant_pos_vec,
                mutant_bool_ballot_matrix,
            ),
        )


@lru_cache(maxsize=None)
def build_section_list(m: int, L: int) -> list[int]:
    """
    Helper function listing the number of ballots with prescribed lengths and number of candidates.

    Similar to the _child_block_size Peter is adding to votekit.utils.

    Args:
        m: (int) number of candidates that the permutation is picking from.
        L: (int) maximum length of the permutation.

    Returns:
        section_list (list[int]): list where entry i corresponds to the number of permutations of m
            items with length at most L where the first i positions have been prescribed.
            Includes the empty permutation as an option (at every depth).
    """
    section = 1
    section_list = [1]
    for k in range(L - 1, -1, -1):
        section = section * (m - k) + 1
        section_list.append(section)
    section_list.reverse()
    return section_list


def _vectorized_perm_updater(
    winner_comb_vec: NDArray, m: int, L: int, winner_bitsring_vec: NDArray, winner_vec: NDArray
):
    """
    Vectorized updater for a winner combination vec when new winners are added in each position.

    For each index in winner_comb_vec, computes the index of the new winner combination
        that results from adding the new winner specified in winner_vec.
    This updater never constructs explicit permutations and uses the section_list instead.

    Args:
        winner_comb_vec: (NDArray) the previous winner combination indices in need of update.
        m: (int) the number of candidates.
        L: (int) the maximum ranking length.
        winner_bitsring_vec: (NDArray) the previous winner bitstring for each ballot,
        winner_vec: (NDArray) the new winners to be added for each permutation.
    """
    winner_mask_array = np.left_shift(1, winner_vec.astype(np.int64)) - 1
    truncated_winner_mask_array = np.bitwise_and(winner_mask_array, winner_bitsring_vec)
    no_update_needed = np.bitwise_and(winner_mask_array + 1, winner_bitsring_vec) != 0
    update_needed = ~no_update_needed
    if np.any(no_update_needed):
        raise ValueError(
            f"skipping update for winner_vec indices {np.where(no_update_needed)[0]} since candidate is already in the winner combination.\n winner_vec: {winner_vec}\n winner_bitstring_vec: {winner_bitsring_vec}\n winner_mask_array: {winner_mask_array}"
        )
    sections = build_section_list(m, L)
    L_vec = np.bitwise_count(winner_bitsring_vec[update_needed])
    section_vec = np.array(sections)[L_vec + 1]
    shift_vec = np.bitwise_count(truncated_winner_mask_array[update_needed])
    return winner_comb_vec + section_vec * (winner_vec - shift_vec) + 1, np.bitwise_or(
        winner_bitsring_vec, winner_mask_array + 1
    )


def _permutation_matrix_constructor(
    m: int,
    L: int,
    sections: list[int] | None = None,
    dtype: DTypeLike | None = None,
):
    """
    Return a dense matrix where each row is a permutation of m items with length at most L.

    Row i of this matrix should correspond to
        `index_to_lexicographic_ballot(index: i-1, n_candidates: m, max_length: L)`
        from votekit.utils.
    Building this matrix is a bad idea when m, L are greater than 10.

    Args:
        m: (int) number of candidates that each permutation is picking from.
        L: (int) the maximum ranking length of each permutation.
        sections: (list[int] | None) pre-computed section list for the given m, L.
            If None, this will be computed by the function.
        dtype: (np.dtype | None) the dtype of the output array. If None, this will be int8 if possible
            and int16 otherwise.
    """
    if sections is None:
        sections = build_section_list(m, L)

    if len(sections) != L + 1:
        raise ValueError("sections must have length L+1")

    if dtype is None:
        out_dtype: np.dtype[Any] = np.dtype(np.int16 if m > np.iinfo(np.int8).max else np.int8)
    else:
        out_dtype = np.dtype(dtype)

    A = np.full((sections[0], L), -1, dtype=out_dtype)
    used = np.zeros(m, dtype=bool)

    def fill(start, depth):
        if depth == L:
            return

        section = sections[depth + 1]
        row = start + 1

        for x in range(m):
            if used[x]:
                continue

            A[row : row + section, depth] = x
            used[x] = True
            fill(row, depth + 1)
            used[x] = False

            row += section

    fill(0, 0)
    return A
