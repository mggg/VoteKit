from votekit.elections.election_types.ranking.abstract_ranking import RankingElection
from votekit.elections.transfers import fractional_transfer
from votekit.elections.election_types.ranking.stv.utils import numpy_random_transfer
from votekit.pref_profile import RankProfile, ProfileError
from votekit.elections.election_state import ElectionState
from votekit.ballot import RankBallot
from votekit.cleaning import (
    remove_and_condense_rank_profile,
    remove_cand_rank_ballot,
    condense_rank_ballot,
)
from votekit.utils import (
    _first_place_votes_from_df_no_ties,
    ballots_by_first_cand,
    tiebreak_set,
    elect_cands_from_set_ranking,
    score_dict_to_ranking,
)
from votekit.elections.election_types.ranking.stv.numpy_stv_base import (
    NumpySTVBase,
    NumpyElectionDataTracker,
    NumpySTVSentinel,
    QuotaType,
    TiebreakType,
    TransferType,
    ElectionPlay,
)

from typing import Callable, Union
import numpy as np
from numpy.typing import NDArray
from warnings import warn


class NumpyInnerSTV(NumpySTVBase):
    """
    Most general version of an STV election.

    Contains niche arguments such as "dynamic_threshold" that are not exposed in the main STV
        class.
    """

    def __init__(
        self,
        profile: RankProfile,
        m=1,
        transfer: TransferType = "fractional",
        quota: QuotaType | None = "droop",
        simultaneous: bool | None = True,
        tiebreak: TiebreakType | None = None,
        dynamic_threshold: bool = False,
        block_rcv: bool = False,
    ):
        """
        Initialize an STV election with advanced options.

        Args:
            profile (RankProfile): RankProfile to run election on.
            m (int): Number of seats to be elected. Defaults to 1.
            transfer (TransferType, optional): Transfer method to be used. Accepts "fractional", "fractional_random", "cambridge_random", and "random".
                Defaults to "fractional".
            quota (QuotaType, optional): Formula to calculate quota. Accepts "droop" or "hare".
                Defaults to "droop".
            simultaneous (bool, optional): True if all candidates who cross threshold in a round
                are elected simultaneously. False if only the candidate with highest first-place
                votes who crosses the threshold is elected in a round. Defaults to True.
            tiebreak (TiebreakType | None, optional): Method to be used if a tiebreak is needed. Accepts
                "borda", "random", and "cambridge_random". Defaults to None, in which case a ValueError is raised if
                a tiebreak is needed.
            dynamic_threshold (bool, optional): If True, threshold is recalculated each round based on
                the number of remaining active votes. Defaults to False.
            block_rcv (bool, optional): If True, blocks ranked-choice voting. Defaults to False.
        """
        self.__check_profile_and_seats_and_candidates_and_transfer(profile, m, transfer)
        super().__init__(
            profile=profile,
            m=m,
            tiebreak=tiebreak,
        )
        self.transfer = transfer if transfer != "random" else "cambridge_random"
        self.quota = quota
        self.simultaneous = simultaneous
        self.dynamic_threshold = dynamic_threshold
        self.block_rcv = block_rcv
        self.threshold = self._get_threshold(quota, float(np.sum(self._data.wt_vec)))
        self._run_and_store()

    def __check_profile_and_seats_and_candidates_and_transfer(
        self, profile: RankProfile, m: int, transfer: TransferType
    ):
        """
        Initial validation of the arguments passed to the STV election.

        Does the following:
            - Checks if the profile is a RankProfile,
            - Checks if the number of seats is positive,
            - Checks if there are enough candidates to fill the seats,
            - Checks if the transfer method is implemented.
            - Warns the user that the "random" transfer is ambiguous if chosen.

        Args:
            profile (RankProfile): The preference profile to validate.
            m (int): The number of seats to be elected.
            transfer (TransferType): The transfer method to be used.
        """
        if not isinstance(profile, RankProfile):
            raise ProfileError("Profile must be of type RankProfile.")
        if m <= 0:
            raise ValueError("m must be positive.")
        elif len(profile.candidates_cast) < m:
            raise ValueError("Not enough candidates received votes to be elected.")
        if transfer not in [
            "fractional",
            "cambridge_random",
            "fractional_random",
            "random",
        ]:
            raise ValueError(
                "Transfer method must be either 'fractional', 'cambridge_random', 'fractional_random', or 'random'."
            )
        if transfer == "random":
            warn(
                "The 'random' transfer method is ambiguous, and being interpreted as 'cambridge_random'. "
                "Please specify 'cambridge_random' or 'fractional_random' to avoid this warning."
            )

    def _update_because_winner(
        self,
        winners: list[int],
        tallies: NDArray,
        quota: float,
        mutated_fpv_vec: NDArray,
        mutated_wt_vec: NDArray,
        bool_ballot_matrix: NDArray,
        mutated_pos_vec: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Updates helper arrays after candidates have been elected, transferring surplus votes.

        This method handles the vote transfer process when one or more candidates cross the
        election threshold. It moves ballot pointers to the next available candidate and
        adjusts vote weights according to the transfer method (fractional or random).

        Args:
            winners (list[int]): List of candidate indices who were elected this round.
            tallies (NDArray): Current vote tallies for all candidates.
            mutated_fpv_vec (NDArray): First preference vector (modified in place).
            mutated_wt_vec (NDArray): Weight vector for ballots (modified in place).
            bool_ballot_matrix (NDArray): Boolean mask indicating entries of the ballot matrix
                not eliminated or exhausted.
            mutated_pos_vec (NDArray): Position vector tracking current ballot positions.

        Returns:
            tuple[NDArray, NDArray, NDArray, NDArray]: Updated helper arrays:
                (mutated_fpv_vec, mutated_wt_vec, bool_ballot_matrix, mutated_pos_vec).
        """
        rows_with_winner_fpv = np.isin(mutated_fpv_vec, winners)
        winner_row_indices = np.where(rows_with_winner_fpv)[0]
        allowed_pos_matrix = bool_ballot_matrix[winner_row_indices]

        next_fpv_pos_vec = allowed_pos_matrix.argmax(axis=1)
        mutated_pos_vec[winner_row_indices] = next_fpv_pos_vec

        next_fpv_vec = self._data.ballot_matrix[winner_row_indices, next_fpv_pos_vec]

        if self.block_rcv:
            mutated_fpv_vec[rows_with_winner_fpv] = next_fpv_vec
            return (
                mutated_fpv_vec,
                mutated_wt_vec,
                bool_ballot_matrix,
                mutated_pos_vec,
            )
        if self.transfer == "fractional":
            get_transfer_value_vec = np.frompyfunc(
                lambda w: (tallies[w] - quota) / tallies[w], 1, 1
            )
            transfer_value_vec = get_transfer_value_vec(
                mutated_fpv_vec[rows_with_winner_fpv]
            ).astype(np.float64)
            mutated_wt_vec[rows_with_winner_fpv] *= transfer_value_vec
            mutated_fpv_vec[rows_with_winner_fpv] = next_fpv_vec
        elif self.transfer is not None and "random" in self.transfer:
            new_weights = np.zeros_like(mutated_wt_vec, dtype=np.int64)
            for winner_idx in winners:
                if self.transfer == "cambridge_random":
                    mutated_fpv_vec[
                        winner_row_indices[
                            next_fpv_vec == NumpySTVSentinel.BLANK_RANKING.value
                        ]
                    ] = NumpySTVSentinel.BLANK_RANKING.value
                surplus = int(tallies[winner_idx] - quota)
                counts = numpy_random_transfer(
                    fpv_vec=mutated_fpv_vec,
                    wt_vec=mutated_wt_vec,
                    winner=winner_idx,
                    surplus=surplus,
                )
                new_weights += counts.astype(new_weights.dtype)

            mutated_wt_vec[winner_row_indices] = new_weights[winner_row_indices]
            mutated_fpv_vec[rows_with_winner_fpv] = next_fpv_vec
        return (
            mutated_fpv_vec,
            mutated_wt_vec,
            bool_ballot_matrix,
            mutated_pos_vec,
        )

    def _update_because_loser(
        self,
        loser: int,
        mutated_fpv_vec: NDArray,
        wt_vec: NDArray,
        bool_ballot_matrix: NDArray,
        mutated_pos_vec: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Updates helper arrays after a single candidate has been eliminated.

        There's not a lot to do here -- find_loser already updates the stencil, so we just
        need to move the position and FPV vectors to their pre-calculated new positions.

        Args:
            loser (int): Index of the candidate who was eliminated this round.
            mutated_fpv_vec (NDArray): First preference vector (modified in place).
            wt_vec (NDArray): Weight vector for ballots (modified in place).
            bool_ballot_matrix (NDArray): Boolean mask indicating entries of the ballot matrix
                not eliminated or exhausted.
            mutated_pos_vec (NDArray): Position vector tracking current ballot positions.

        Returns:
            tuple[NDArray, NDArray, NDArray, NDArray]: Updated helper arrays:
                (mutated_fpv_vec, wt_vec, bool_ballot_matrix, mutated_pos_vec).
        """
        rows_with_loser_fpv = np.isin(mutated_fpv_vec, loser)
        loser_row_indices = np.where(rows_with_loser_fpv)[0]
        allowed_pos_matrix = bool_ballot_matrix[loser_row_indices]
        mutated_pos_vec[loser_row_indices] = allowed_pos_matrix.argmax(axis=1)
        mutated_fpv_vec[loser_row_indices] = self._data.ballot_matrix[
            loser_row_indices, mutated_pos_vec[loser_row_indices]
        ]

        return (
            mutated_fpv_vec,
            wt_vec,
            bool_ballot_matrix,
            mutated_pos_vec,
        )

    def _find_loser(
        self,
        tallies: NDArray,
        round_number: int,
        mutant_bool_ballot_matrix: NDArray,
        mutant_winner_list: list[int],
        mutant_eliminated_or_exhausted: list[int],
        mutant_tiebreak_record: list[dict[frozenset[str], tuple[frozenset[str], ...]]],
    ) -> tuple[
        int,
        tuple[
            NDArray,
            list[int],
            list[int],
            list[dict[frozenset[str], tuple[frozenset[str], ...]]],
        ],
    ]:
        """
        Identify the candidate to eliminate in the current round, applying tiebreaks if necessary.

        Args:
            tallies (NDArray): Current tallies for each candidate.
            round_number (int): The current round number.
            mutant_bool_ballot_matrix (NDArray): Boolean mask for eliminated candidates.
            mutant_winner_list (list[int]): List of winner candidate indices so far.
            mutant_eliminated_or_exhausted (list[int]): List of eliminated candidate indices so far.
            mutant_tiebreak_record (list[dict[frozenset[str], tuple[frozenset[str], ...]]]):
                Tiebreak record for each round.

        Returns:
            tuple: (index of eliminated candidate, updated state tuple containing the boolean
                ballot matrix, winner list, eliminated list, and tiebreak record).
        """
        masked_tallies: NDArray = np.where(
            np.isin(np.arange(len(tallies)), mutant_eliminated_or_exhausted),
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
        mutant_eliminated_or_exhausted.append(loser_idx)
        self._update_bool_ballot_matrix(mutant_bool_ballot_matrix, [loser_idx])
        return loser_idx, (
            mutant_bool_ballot_matrix,
            mutant_winner_list,
            mutant_eliminated_or_exhausted,
            mutant_tiebreak_record,
        )

    def _find_winners(
        self,
        tallies: NDArray,
        round_number: int,
        quota: float,
        mutant_bool_ballot_matrix: NDArray,
        mutant_winner_list: list[int],
        mutant_eliminated_or_exhausted: list[int],
        mutant_tiebreak_record: list[dict[frozenset[str], tuple[frozenset[str], ...]]],
    ) -> tuple[
        list[int],
        tuple[
            NDArray,
            list[int],
            list[int],
            list[dict[frozenset[str], tuple[frozenset[str], ...]]],
        ],
    ]:
        """
        Identify the candidate(s) to elect in the current round, applying tiebreaks if necessary.

        Args:
            tallies (NDArray): Current tallies for each candidate.
            round_number (int): The current round number.
            quota (float): Threshold required for election in this round.
            mutant_bool_ballot_matrix (NDArray): Boolean mask for eliminated/elected candidates.
            mutant_winner_list (list[int]): List of winner candidate indices so far.
            mutant_eliminated_or_exhausted (list[int]): List of eliminated/elected candidate
                indices so far.
            mutant_tiebreak_record (list[dict[frozenset[str], tuple[frozenset[str], ...]]]):
                Tiebreak record for each round.

        Returns:
            tuple: (list of elected candidate indices, updated state tuple containing the boolean
                ballot matrix, winner list, eliminated list, and tiebreak record).
        """
        if self.simultaneous:
            winners_temp = np.where(tallies >= quota)[0]
            winners_temp = winners_temp[np.argsort(-tallies[winners_temp])]
            winners = winners_temp.tolist()
            mutant_tiebreak_record.append({})
        else:
            if np.count_nonzero(tallies == np.max(tallies)) > 1:
                potential_winners: list[int] = (
                    np.where(tallies == tallies.max())[0].astype(int).tolist()
                )
                winner_idx, mutant_tiebreak_record = self._run_winner_tiebreak(
                    potential_winners, round_number, mutant_tiebreak_record
                )
            else:
                winner_idx = int(np.argmax(tallies))
                mutant_tiebreak_record.append({})
            winners = [winner_idx]
        for winner_idx in winners:
            mutant_winner_list.append(int(winner_idx))
            mutant_eliminated_or_exhausted.append(winner_idx)
        self._update_bool_ballot_matrix(mutant_bool_ballot_matrix, winners)
        return winners, (
            mutant_bool_ballot_matrix,
            mutant_winner_list,
            mutant_eliminated_or_exhausted,
            mutant_tiebreak_record,
        )

    def _update_bool_ballot_matrix(
        self, _mutant_bool_ballot_matrix: NDArray, newly_gone: list[int]
    ) -> NDArray:
        """
        Update the stencil mask to mark candidates as eliminated or elected.

        Args:
            _mutant_bool_ballot_matrix (NDArray): Boolean mask of eliminated/elected candidates.
            newly_gone (list[int]): List of candidate indices to mark as eliminated/elected.

        Returns:
            NDArray: Updated stencil mask.
        """
        _mutant_bool_ballot_matrix &= ~np.isin(self._data.ballot_matrix, newly_gone)
        return _mutant_bool_ballot_matrix

    def _run_election(self, data: NumpyElectionDataTracker) -> tuple[
        list[NDArray],
        list[ElectionPlay],
        list[dict[frozenset[str], tuple[frozenset[str], ...]]],
    ]:
        """
        Core election logic for STV.

        Args:
            data (NumpyElectionDataTracker): The initialized data tracker with the profile
                converted to numpy arrays.

        Returns:
            fpv_by_round (list[NDArray]): List of first-preference vote tallies by round.
            play_by_play (list[ElectionPlay]): List of dictionaries representing the actions
                 taken in each round.
            tiebreak_record (list[dict[frozenset[str], tuple[frozenset[str], ...]]]):
                List of dictionaries representing tiebreak resolutions for each round.
        """
        ballot_matrix = data.ballot_matrix
        wt_vec = np.copy(data.wt_vec)
        fpv_vec = np.copy(ballot_matrix[:, 0])
        m = self.m
        ncands = len(self.candidates)

        fpv_scores_by_round = []
        play_by_play: list[ElectionPlay] = []
        round_number = 0
        quota = self.threshold
        ballot_weight_sitting_with_winners = 0.0
        winner_list: list[int] = []
        eliminated_or_exhausted: list[int] = []
        tiebreak_record: list[dict[frozenset[str], tuple[frozenset[str], ...]]] = []
        pos_vec: NDArray = np.zeros(ballot_matrix.shape[0], dtype=np.int8)
        mutant_bool_ballot_matrix: NDArray = np.ones_like(ballot_matrix, dtype=bool)

        def make_tallies(fpv_vec: NDArray, wt_vec: NDArray, ncands: int) -> NDArray:
            """
            Compute weighted first-preference tallies.
            """
            return np.bincount(
                fpv_vec[fpv_vec != NumpySTVSentinel.BLANK_RANKING.value],
                weights=wt_vec[fpv_vec != NumpySTVSentinel.BLANK_RANKING.value],
                minlength=ncands,
            )

        mutant_engine = (fpv_vec, wt_vec, mutant_bool_ballot_matrix, pos_vec)
        mutant_record = (
            mutant_bool_ballot_matrix,
            winner_list,
            eliminated_or_exhausted,
            tiebreak_record,
        )
        while len(winner_list) < m:
            tallies = make_tallies(fpv_vec, wt_vec, ncands)
            if self.dynamic_threshold:
                quota = self._get_threshold(
                    self.quota, tallies.sum() + ballot_weight_sitting_with_winners
                )
            fpv_scores_by_round.append(tallies.copy())
            while np.any(tallies >= quota):
                winners, mutant_record = self._find_winners(
                    tallies, round_number, quota, *mutant_record
                )
                mutant_engine = self._update_because_winner(
                    winners, tallies, quota, *mutant_engine
                )
                play_by_play.append(
                    ElectionPlay(
                        round_number=int(round_number),
                        winners=[int(c) for c in winners],
                        wt_vec=mutant_engine[1].copy(),
                        round_type="election",
                    )
                )
                round_number += 1
                tallies = make_tallies(fpv_vec, wt_vec, ncands)
                if self.dynamic_threshold:
                    play_by_play[-1]["threshold"] = float(quota)
                    ballot_weight_sitting_with_winners += len(winners) * float(quota)
                    quota = self._get_threshold(
                        self.quota, tallies.sum() + ballot_weight_sitting_with_winners
                    )
                fpv_scores_by_round.append(tallies.copy())
            if len(winner_list) == m:
                break
            if len(eliminated_or_exhausted) - len(winner_list) == ncands - m:
                still_standing = [
                    int(i) for i in range(ncands) if i not in eliminated_or_exhausted
                ]
                winner_list += still_standing
                play_by_play.append(
                    ElectionPlay(
                        round_number=int(round_number),
                        winners=still_standing,
                        wt_vec=np.zeros_like(fpv_vec, dtype=np.float64),
                        round_type="default",
                    )
                )
                if self.dynamic_threshold:
                    play_by_play[-1]["threshold"] = float(quota)
                round_number += 1
                fpv_scores_by_round.append(np.zeros(ncands, dtype=np.float64))
                tiebreak_record.append({})
                break
            loser_idx, mutant_record = self._find_loser(
                tallies, round_number, *mutant_record
            )
            mutant_engine = self._update_because_loser(loser_idx, *mutant_engine)
            play_by_play.append(
                ElectionPlay(
                    round_number=int(round_number),
                    loser=[int(loser_idx)],
                    round_type="elimination",
                )
            )
            if self.dynamic_threshold:
                play_by_play[-1]["threshold"] = float(quota)
            round_number += 1
        return fpv_scores_by_round, play_by_play, tiebreak_record


class FastSTV(NumpyInnerSTV):
    """
    STV elections. Must be given a RankProfile.
    """

    def __init__(
        self,
        profile: RankProfile,
        m: int = 1,
        transfer: TransferType = "fractional",
        quota: QuotaType | None = "droop",
        simultaneous: bool = True,
        tiebreak: TiebreakType | None = None,
    ):
        """
        Initialize a fast STV election.

        Args:
            profile (RankProfile): RankProfile to run election on.
            m (int, optional): Number of seats to be elected. Defaults to 1.
            transfer (TransferType, optional): Transfer method to be used. Accepts "fractional", "random",
                "fractional_random", and "cambridge_random". Defaults to "fractional".
            quota (QuotaType, optional): Formula to calculate quota. Accepts "droop" or "hare".
                Defaults to "droop".
            simultaneous (bool, optional): True if all candidates who cross threshold in a round are
                elected simultaneously. False if only the candidate with highest first-place
                votes who crosses the threshold is elected in a round. Defaults to True.
            tiebreak (TiebreakType | None, optional): Method to be used if a tiebreak is needed. Accepts
                "borda", "random", and "cambridge_random". Defaults to None, in which case a ValueError is raised if
                a tiebreak is needed.
        """
        super().__init__(
            profile=profile,
            m=m,
            transfer=transfer,
            quota=quota,
            simultaneous=simultaneous,
            tiebreak=tiebreak,
        )


class AlbanySTV(NumpyInnerSTV):
    """
    STV variant used in Albany, CA.

    Differs from FastSTV in that the threshold is recalculated each round based on remaining
        votes and candidates.
    """

    def __init__(
        self,
        profile: RankProfile,
        m: int = 1,
        transfer: TransferType = "fractional",
        quota: QuotaType | None = "droop",
        simultaneous: bool = True,
        tiebreak: TiebreakType | None = None,
    ):
        """
        Initialize an Albany STV election.

        Args:
            profile (RankProfile): RankProfile to run election on.
            m (int, optional): Number of seats to be elected. Defaults to 1.
            transfer (TransferType, optional): Transfer method to be used. Accepts "fractional", "random",
                "fractional_random", and "cambridge_random". Defaults to "fractional".
            quota (QuotaType, optional): Formula to calculate quota. Accepts "droop" or "hare".
                Defaults to "droop".
            simultaneous (bool, optional): True if all candidates who cross threshold in a round are
                elected simultaneously. False if only the candidate with highest first-place
                votes who crosses the threshold is elected in a round. Defaults to True.
            tiebreak (TiebreakType | None, optional): Method to be used if a tiebreak is needed. Accepts
                "borda", "random", and "cambridge_random". Defaults to None, in which case a ValueError is raised if
                a tiebreak is needed.
        """
        super().__init__(
            profile=profile,
            m=m,
            transfer=transfer,
            quota=quota,
            simultaneous=simultaneous,
            tiebreak=tiebreak,
            dynamic_threshold=True,
        )


class FastIRV(NumpyInnerSTV):
    """
    Elect exactly 1 seat using IRV (Instant-runoff voting) elections.

    All ballots must have no ties. Equivalent to STV for m = 1.
    """

    def __init__(
        self,
        profile: RankProfile,
        quota: QuotaType | None = "droop",
        tiebreak: TiebreakType | None = None,
    ):
        """
        Initialize a fast IRV election.

        Args:
            profile (RankProfile): RankProfile to run election on.
            quota (QuotaType, optional): Formula to calculate quota. Accepts "droop" or "hare".
                Defaults to "droop".
            tiebreak (TiebreakType | None, optional): Method to be used if a tiebreak is needed. Accepts
                "borda", "random", and "cambridge_random". Defaults to None, in which case a ValueError is raised if
                a tiebreak is needed.
        """
        super().__init__(
            profile=profile,
            m=1,
            transfer="fractional",
            quota=quota,
            tiebreak=tiebreak,
        )


class FastSequentialRCV(NumpyInnerSTV):
    """
    STV election in which votes are not transferred from elected candidates.

    This system just runs a series of IRV elections until the desired number of candidates are elected.

    Notes:
     - Used in parts of Utah.
    """

    def __init__(
        self,
        profile: RankProfile,
        m: int | None = 1,
        quota: QuotaType | None = "droop",
        simultaneous: bool | None = True,
        tiebreak: TiebreakType | None = None,
    ):
        """
        Initialize a fast sequential RCV election.

        Args:
            profile (RankProfile): RankProfile to run election on.
            m (int, optional): Number of seats to be elected. Defaults to 1.
            quota (QuotaType, optional): Formula to calculate quota. Accepts "droop" or "hare".
                Defaults to "droop".
            simultaneous (bool, optional): True if all candidates who cross threshold in a round are
                elected simultaneously. False if only the candidate with highest first-place
                votes who crosses the threshold is elected in a round. Defaults to True.
            tiebreak (TiebreakType | None, optional): Method to be used if a tiebreak is needed. Accepts
                "borda", "random", and "cambridge_random". Defaults to None, in which case a ValueError is raised if
                a tiebreak is needed.
        """
        super().__init__(
            profile=profile,
            m=m,
            quota=quota,
            simultaneous=simultaneous,
            tiebreak=tiebreak,
            block_rcv=True,
        )


class STV(RankingElection):
    """
    STV elections. All ballots must have no ties.
    """

    def __init__(
        self,
        profile: RankProfile,
        m: int = 1,
        transfer: Callable[
            [str, float, Union[tuple[RankBallot], list[RankBallot]], int],
            tuple[RankBallot, ...],
        ] = fractional_transfer,
        quota: QuotaType | None = "droop",
        simultaneous: bool = True,
        tiebreak: TiebreakType | None = None,
    ):
        """
        Initialize an STV election.

        Args:
            profile (RankProfile): RankProfile to run election on.
            m (int): Number of seats to be elected. Defaults to 1.
            transfer (Callable[[str, float, Union[tuple[RankBallot], list[RankBallot]], int],
                tuple[RankBallot, ...]]): Transfer method. Defaults to fractional transfer.
                Function signature is elected candidate, their number of first-place votes, the list
                of ballots with them ranked first, and the threshold value. Returns the list of
                ballots after transfer.
            quota (QuotaType, optional): Formula to calculate quota. Accepts "droop" or "hare".
                Defaults to "droop".
            simultaneous (bool, optional): True if all candidates who cross threshold in a round are
                elected simultaneously. False if only the candidate with highest first-place votes
                who crosses the threshold is elected in a round. Defaults to True.
            tiebreak (TiebreakType | None, optional): Method to be used if a tiebreak is needed. Accepts
                "borda" and "random". Defaults to None, in which case a ValueError is raised if
                a tiebreak is needed.
        """
        self._stv_validate_profile(profile)

        if m <= 0:
            raise ValueError("m must be positive.")
        elif len(profile.candidates_cast) < m:
            raise ValueError("Not enough candidates received votes to be elected.")
        self.m = m
        self.transfer = transfer
        self.quota = quota

        self.threshold = 0
        self.threshold = self.get_threshold(profile.total_ballot_wt)
        self.simultaneous = simultaneous
        self.tiebreak = tiebreak

        super().__init__(
            profile,
            score_function=_first_place_votes_from_df_no_ties,
            sort_high_low=True,
        )

    def _stv_validate_profile(self, profile: RankProfile):
        """
        Validate that each ballot has a ranking, and that there are no ties in ballots.

        Args:
            profile (RankProfile): Profile to validate.
        """
        if not isinstance(profile, RankProfile):
            raise ProfileError("Profile must be of type RankProfile.")
        assert profile.max_ranking_length is not None
        ranking_rows = [
            f"Ranking_{i}" for i in range(1, profile.max_ranking_length + 1)
        ]
        try:
            np_arr = profile.df[ranking_rows].to_numpy()
            weight_col = profile.df["Weight"]
        except KeyError:
            raise TypeError("Ballots must have rankings.")

        tilde = frozenset({"~"})
        for idx, row in enumerate(np_arr):
            if any(len(s) > 1 for s in row):
                raise TypeError(
                    f"Ballot {RankBallot(ranking=tuple(row.tolist()), weight = weight_col[idx])} "
                    "contains a tied ranking."
                )
            if (row == tilde).all():
                raise TypeError("Ballots must have rankings.")

    def get_threshold(self, total_ballot_wt: float) -> int:
        """
        Calculates threshold required for election.

        Args:
            total_ballot_wt (float): Total weight of ballots to compute threshold.

        Returns:
            int: Value of the threshold.
        """
        if self.threshold == 0:
            if self.quota == "droop":
                return int(total_ballot_wt / (self.m + 1) + 1)  # takes floor
            elif self.quota == "hare":
                return int(total_ballot_wt / self.m)  # takes floor
            else:
                raise ValueError("Misspelled or unknown quota type.")
        else:
            return self.threshold

    def _is_finished(self):
        """
        Return True if all seats have been filled.
        """
        elected_cands = [c for s in self.get_elected() for c in s]

        if len(elected_cands) == self.m:
            return True
        return False

    def _simultaneous_elect_step(
        self, profile: RankProfile, prev_state: ElectionState
    ) -> tuple[tuple[frozenset[str], ...], RankProfile]:
        """
        Run one step of an election from the given profile and previous state.

        Used for simultaneous STV election if candidates cross threshold.

        Args:
            profile (RankProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.

        Returns:
            tuple[tuple[frozenset[str],...], RankProfile]:
                A tuple whose first entry is the elected candidates, ranked by first-place votes,
                and whose second entry is the profile of ballots after transfers.
        """
        ranking_by_fpv = prev_state.remaining
        current_round = prev_state.round_number + 1

        elected = []
        if current_round < len(self.election_states):
            elected = list(self.election_states[current_round].elected)
        else:
            for s in ranking_by_fpv:
                c = list(s)[0]  # all cands in set have same score
                if prev_state.scores[c] >= self.threshold:
                    elected.append(s)

                # since ranking is ordered by fpv, once below threshold we are done
                else:
                    break

        ballots_by_fpv = ballots_by_first_cand(profile)
        new_ballots = [RankBallot()] * profile.num_ballots
        ballot_index = 0

        for s in elected:
            for candidate in s:
                transfer_ballots = self.transfer(
                    candidate,
                    prev_state.scores[candidate],
                    ballots_by_fpv[candidate],
                    self.threshold,
                )
                new_ballots[ballot_index : (ballot_index + len(transfer_ballots))] = (
                    transfer_ballots
                )
                ballot_index += len(transfer_ballots)

        for candidate in set([c for s in ranking_by_fpv for c in s]).difference(
            [c for s in elected for c in s]
        ):
            transfer_ballots = tuple(ballots_by_fpv[candidate])
            new_ballots[ballot_index : (ballot_index + len(transfer_ballots))] = (
                transfer_ballots
            )
            ballot_index += len(transfer_ballots)

        cleaned_ballots = tuple(
            condense_rank_ballot(
                remove_cand_rank_ballot([c for s in elected for c in s], b)
            )
            for b in new_ballots
            if b.ranking
        )

        remaining_cands = set(profile.candidates_cast).difference(
            [c for s in elected for c in s]
        )

        new_profile = RankProfile(
            ballots=cleaned_ballots,
            candidates=tuple(remaining_cands),
            max_ranking_length=profile.max_ranking_length,
        )
        return (tuple(elected), new_profile)

    def _single_elect_step(
        self, profile: RankProfile, prev_state: ElectionState
    ) -> tuple[
        tuple[frozenset[str], ...],
        dict[frozenset[str], tuple[frozenset[str], ...]],
        RankProfile,
    ]:
        """
        Run one step of an election from the given profile and previous state.

        Used for one-by-one STV election if candidates cross threshold.

        Args:
            profile (RankProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.

        Returns:
            tuple[tuple[frozenset[str],...], dict[frozenset[str], tuple[frozenset[str],...]],
            RankProfile]:
                A tuple whose first entry is the elected candidate, second is the tiebreak dict,
                and whose third entry is the profile of ballots after transfers.
        """
        ranking_by_fpv = prev_state.remaining
        current_round = prev_state.round_number + 1

        if current_round < len(self.election_states):
            elected = self.election_states[current_round].elected
            remaining = self.election_states[current_round].remaining
            tiebreaks = self.election_states[current_round].tiebreaks
        else:
            elected, remaining, tiebreak = elect_cands_from_set_ranking(
                ranking_by_fpv, m=1, profile=profile, tiebreak=self.tiebreak
            )
            if tiebreak:
                tiebreaks = {tiebreak[0]: tiebreak[1]}
            else:
                tiebreaks = {}

        ballots_by_fpv = ballots_by_first_cand(profile)
        new_ballots = [RankBallot()] * profile.num_ballots
        ballot_index = 0

        elected_c = list(elected[0])[0]

        transfer_ballots = self.transfer(
            elected_c,
            prev_state.scores[elected_c],
            ballots_by_fpv[elected_c],
            self.threshold,
        )
        new_ballots[ballot_index : (ballot_index + len(transfer_ballots))] = (
            transfer_ballots
        )
        ballot_index += len(transfer_ballots)

        for s in remaining:
            for candidate in s:
                transfer_ballots = tuple(ballots_by_fpv[candidate])
                new_ballots[ballot_index : (ballot_index + len(transfer_ballots))] = (
                    transfer_ballots
                )
                ballot_index += len(transfer_ballots)

        cleaned_ballots = tuple(
            condense_rank_ballot(remove_cand_rank_ballot(elected_c, b))
            for b in new_ballots
            if b.ranking
        )

        remaining_cands = set(profile.candidates_cast).difference(
            [c for s in elected for c in s]
        )
        new_profile = RankProfile(
            ballots=cleaned_ballots,
            candidates=tuple(remaining_cands),
            max_ranking_length=profile.max_ranking_length,
        )
        return elected, tiebreaks, new_profile

    def _run_step(
        self, profile: RankProfile, prev_state: ElectionState, store_states=False
    ) -> RankProfile:
        """
        Run one step of an election from the given profile and previous state.

        STV sets a threshold for first-place votes. If a candidate passes it, they are elected.
        We remove them from all ballots and transfer any surplus ballots to other candidates.
        If no one passes, we eliminate the lowest ranked candidate and reallocate their ballots.

        Can be run 1-by-1 or simultaneous, which determines what happens if multiple people cross
        threshold.

        Args:
            profile (RankProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.
            store_states (bool, optional): True if `self.election_states` should be updated with the
                ElectionState generated by this round. This should only be True when used by
                `self._run_election()`. Defaults to False.

        Returns:
            RankProfile: The profile of ballots after the round is completed.
        """

        tiebreaks: dict[frozenset[str], tuple[frozenset[str], ...]] = {}

        current_round = prev_state.round_number + 1
        above_thresh_cands = [
            c for c, score in prev_state.scores.items() if score >= self.threshold
        ]

        if len(above_thresh_cands) > 0:
            if self.simultaneous:
                elected, new_profile = self._simultaneous_elect_step(
                    profile, prev_state
                )

            else:
                elected, tiebreaks, new_profile = self._single_elect_step(
                    profile, prev_state
                )
            # no one eliminated in elect round
            eliminated: tuple[frozenset[str], ...] = (frozenset(),)

        # catches the possibility that we exhaust all ballots
        # without candidates reaching threshold

        elif len(profile.candidates_cast) == self.m - len(
            [c for s in self.get_elected() for c in s]
        ):
            elected = prev_state.remaining
            eliminated = (frozenset(),)
            new_profile = RankProfile()

        else:
            lowest_fpv_cands = prev_state.remaining[-1]

            if len(lowest_fpv_cands) > 1:
                tiebroken_ranking = None
                if current_round < len(self.election_states):
                    possible_tiebreaks = list(
                        self.election_states[current_round].tiebreaks.values()
                    )
                    if len(possible_tiebreaks) > 0:
                        tiebroken_ranking = possible_tiebreaks[0]
                if tiebroken_ranking is None or len(tiebroken_ranking) == 0:
                    tiebroken_ranking = tiebreak_set(
                        lowest_fpv_cands, self.get_profile(0), tiebreak="first_place"
                    )

                tiebreaks = {lowest_fpv_cands: tiebroken_ranking}

                eliminated_cand = list(tiebroken_ranking[-1])[-1]

            else:
                eliminated_cand = list(lowest_fpv_cands)[0]

            new_profile = remove_and_condense_rank_profile(
                eliminated_cand,
                profile,
                retain_original_candidate_list=False,
            )

            elected = (frozenset(),)
            eliminated = (frozenset([eliminated_cand]),)

        if store_states:
            if self.score_function is None:
                raise ValueError("No score function defined for election.")

            scores = self.score_function(new_profile)

            remaining = score_dict_to_ranking(scores)

            new_state = ElectionState(
                round_number=prev_state.round_number + 1,
                remaining=remaining,
                elected=elected,
                eliminated=eliminated,
                scores=scores,
                tiebreaks=tiebreaks,
            )

            self.election_states.append(new_state)

        return new_profile


class IRV(STV):
    """
    Elect exactly 1 seat using IRV (Instant-runoff voting).

    All ballots must have no ties. Equivalent to STV for m = 1.

    """

    def __init__(
        self,
        profile: RankProfile,
        quota: QuotaType | None = "droop",
        tiebreak: TiebreakType | None = None,
    ):
        """
        Initialize an IRV election.

        Args:
            profile (RankProfile): RankProfile to run election on.
            quota (QuotaType, optional): Formula to calculate quota. Accepts "droop" or "hare".
                Defaults to "droop".
            tiebreak (TiebreakType | None, optional): Method to be used if a tiebreak is needed. Accepts
                "borda" and "random". Defaults to None, in which case a ValueError is raised if
                a tiebreak is needed.
        """
        super().__init__(profile, m=1, quota=quota, tiebreak=tiebreak)


class SequentialRCV(STV):
    """
    STV election in which votes are not transferred from elected candidates.

    This system just runs a series of IRV elections until the desired number of candidates are elected.

    Notes:
     - Used in parts of Utah.

    """

    def __init__(
        self,
        profile: RankProfile,
        m: int = 1,
        quota: QuotaType | None = "droop",
        simultaneous: bool = True,
        tiebreak: TiebreakType | None = None,
    ):
        """
        Initialize a sequential RCV election.

        Args:
            profile (RankProfile): RankProfile to run election on.
            m (int, optional): Number of seats to be elected. Defaults to 1.
            quota (QuotaType, optional): Formula to calculate quota. Accepts "droop" or "hare".
                Defaults to "droop".
            simultaneous (bool, optional): True if all candidates who cross threshold in a round are
                elected simultaneously. False if only the candidate with highest first-place
                votes who crosses the threshold is elected in a round. Defaults to True.
            tiebreak (TiebreakType | None, optional): Method to be used if a tiebreak is needed. Accepts
                "borda" and "random". Defaults to None, in which case a ValueError is raised if
                a tiebreak is needed.
        """

        def _transfer(
            winner: str,
            _fpv: float,
            ballots: Union[tuple[RankBallot], list[RankBallot]],
            _threshold: int,
        ) -> tuple[RankBallot, ...]:
            """
            Transfer ballots by removing the winner and condensing rankings.

            Args:
                winner (str): The candidate to remove from ballots.
                _fpv (float): The number of first-place votes the winner had.
                ballots (Union[tuple[RankBallot], list[RankBallot]]): The ballots to transfer.
                _threshold (int): The threshold for election in this round.

            Returns:
                tuple[RankBallot, ...]: The transferred ballots after removing the winner and condensing rankings.
            """
            del _fpv, _threshold  # unused and del on atomics is okay
            return tuple(
                condense_rank_ballot(remove_cand_rank_ballot(winner, b))
                for b in ballots
            )

        super().__init__(
            profile,
            m=m,
            transfer=_transfer,
            quota=quota,
            simultaneous=simultaneous,
            tiebreak=tiebreak,
        )
