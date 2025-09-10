from .abstract_ranking import RankingElection
from ...transfers import fractional_transfer
from ....pref_profile import PreferenceProfile
from ...election_state import ElectionState
from ....ballot import Ballot
from ....cleaning import (
    remove_and_condense_ranked_profile,
    remove_cand_from_ballot,
    condense_ballot_ranking,
    condense_profile,
)
from ....utils import (
    _first_place_votes_from_df_no_ties,
    ballots_by_first_cand,
    tiebreak_set,
    elect_cands_from_set_ranking,
    score_dict_to_ranking,
    borda_scores,
)
from typing import Optional, Callable, Union
import pandas as pd
import numpy as np
from itertools import groupby


class fastSTV:
    """
    STV elections. All ballots must have no ties.

    Args:
        profile (PreferenceProfile): PreferenceProfile to run election on.
        m (int): Number of seats to be elected. Defaults to 1.
        transfer (Optional[str]): Transfer method to be used. Accepts 'fractional' and 'random'. Defaults to 'fractional'.
        quota (str): Formula to calculate quota. Accepts "droop" or "hare".
            Defaults to "droop".
        simultaneous (bool): True if all candidates who cross threshold in a round are
            elected simultaneously, False if only the candidate with highest first-place votes
            who crosses the threshold is elected in a round. Defaults to False.
        tiebreak (Optional[str]): Method to be used if a tiebreak is needed. Accepts
            'borda' and 'random'. Defaults to None, in which case a ValueError is raised if
            a tiebreak is needed.

    """

    def __init__(
        self,
        profile: PreferenceProfile,
        m: int = 1,
        transfer:str = "fractional",
        quota: str = "droop",
        simultaneous: bool = False,
        tiebreak: Optional[str] = None,
    ):
        self._misc_validation(profile, m, transfer)
        self.profile = profile
        self.m = m
        self.transfer = transfer
        self.quota = quota

        self._ballot_length = profile.max_ranking_length

        self.threshold = self._get_threshold(profile.total_ballot_wt)
        self.simultaneous = simultaneous
        self._winner_tiebreak = tiebreak
        self._loser_tiebreak = tiebreak if tiebreak is not None else 'first_place' #this is what legacy does

        self.candidates = list(profile.candidates)

        self._ballot_matrix, self._wt_vec, self._fpv_vec = self._convert_df(profile)
        self._initial_fpv = self._make_initial_fpv()
        self._fpv_by_round, self._play_by_play, self._tiebreak_record = self._run_STV(
            self._ballot_matrix,
            self._wt_vec.copy(),
            self._fpv_vec,
            m,
            len(self.candidates),
        )
        self.election_states = self._make_election_states()

    def _misc_validation(self, profile: PreferenceProfile, m: int, transfer:str):
        """
        Performs miscellaneous validation checks before running the STV algorithm.

        Args:
            profile (PreferenceProfile): The preference profile to validate.
            m (int): The number of seats to be elected.
            transfer (str): The transfer method to be used.
        """
        if m <= 0:
            raise ValueError("m must be positive.")
        elif len(profile.candidates_cast) < m:
            raise ValueError("Not enough candidates received votes to be elected.")
        self.m = m
        if transfer not in ["fractional", "cambridge_random", "fractional_random"]:
            raise ValueError("Transfer method must be either 'fractional', 'cambridge_random', or 'fractional_random'.")

    def _get_threshold(self, total_ballot_wt: float) -> int:
        """
        Calculates threshold required for election.

        Args:
            total_ballot_wt (float): Total weight of ballots to compute threshold.
        Returns:
            int: Value of the threshold.
        """
        if self.quota == "droop":
            return int(total_ballot_wt / (self.m + 1) + 1)  # takes floor
        elif self.quota == "hare":
            return int(total_ballot_wt / self.m)  # takes floor
        else:
            raise ValueError("Misspelled or unknown quota type.")

    def _convert_df(
        self, profile: PreferenceProfile
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This converts the profile into a numpy matrix with some helper arrays for faster iteration.

        Args:
            profile (PreferenceProfile): The preference profile to convert.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The ballot matrix, weights vector, and first-preference vector.
        """
        df = profile.df
        candidate_to_index = {
            frozenset([name]): i for i, name in enumerate(self.candidates)
        }
        candidate_to_index[frozenset(["~"])] = -127

        ranking_columns = [c for c in df.columns if c.startswith("Ranking")]
        num_rows = len(df)
        num_cols = len(ranking_columns)

        # np matrix of frozensets from the DataFrame
        cells = df[ranking_columns].to_numpy()

        # 1) Convert cells -> np.int8 indices, raising an error if entry not in dict
        def map_cell(cell):
            try:
                return candidate_to_index[cell]
            except KeyError:
                raise TypeError("Ballots must have rankings.")

        # np matrix of indices -- we just need to add padding now
        mapped = np.frompyfunc(map_cell, 1, 1)(cells).astype(np.int8)

        # 2) Add padding
        ballot_matrix = np.full((num_rows, num_cols + 1), -127, dtype=np.int8)
        ballot_matrix[:, :num_cols] = mapped

        # 3) Weights + first-preference vector
        wt_vec = df["Weight"].astype(np.float64).to_numpy()
        fpv_vec = ballot_matrix[:, 0].copy()

        # 4) Reject ballots that have no rankings at all (all -127)
        # Chris thinks this can/should be replaced with pf.contains_rankings == True? 
        empty_rows = np.where(np.all(ballot_matrix == -127, axis=1))[0]
        if empty_rows.size:
            raise TypeError("Ballots must have rankings.")

        return ballot_matrix, wt_vec, fpv_vec
    
    def _make_initial_fpv(self):
        """
        Creates the initial first-preference vote (FPV) vector.

        Returns:
            np.ndarray: The i-th entry is the initial first-preference vote tally for candidate i.
        """
        return np.bincount(
            self._fpv_vec[self._fpv_vec != -127],
            weights=self._wt_vec[self._fpv_vec != -127],
            minlength=len(self.candidates),
        )
    
    def __fpv_tiebreak(self, tied_cands, tiebreak_type) -> tuple[int, tuple[frozenset[str], ...]]:
        """
        Break ties among tied_cands using initial_fpv tallies.

        Args:
            tied_cands (list[int]): List of candidate indices that are tied.
            tiebreak_type (str): Type of tiebreaking to perform ('winner' or 'loser').

        Returns:
            tuple: (chosen_candidate_index, packaged_ranking): the candidate index that won/lost the tiebreak,
                and the packaged tuple of frozensets representing the outcome of the tiebreak.
        """

        tied_cands_set = set(tied_cands)
        if not hasattr(self, '__fpv_clusters'):
            scores = np.asarray(self._initial_fpv)
            order = np.argsort(scores, kind="mergesort")[::-1]
            pairs = [(float(scores[i]), int(i)) for i in order]
            self.__fpv_clusters: list[list[int]] = [
                [idx for _, idx in group]
                for _, group in groupby(pairs, key=lambda x: x[0])
            ]
        
        clusters_containing_tied_cands: list[list[int]] = [[c for c in cluster if c in tied_cands_set]
            for cluster in self.__fpv_clusters if any(i in tied_cands_set for i in cluster)
        ]

        packaged_ranking: tuple[frozenset[str], ...] = tuple(
            frozenset(self.candidates[i] for i in cluster) for cluster in clusters_containing_tied_cands
        )

        relevant = 0 if tiebreak_type == "winner" else -1
        target_cluster = clusters_containing_tied_cands[relevant]

        if len(target_cluster) == 1: #yay
            return target_cluster[0], packaged_ranking

        tiebroken_candidate = int(np.random.choice(target_cluster)) #ok my head is not that empty
        return tiebroken_candidate, packaged_ranking


    def __update_because_winner(
        self,
        winners: list[int],
        tallies: np.ndarray,
        mutated_fpv_vec: np.ndarray,
        mutated_wt_vec: np.ndarray,
        bool_ballot_matrix: np.ndarray,
        mutated_pos_vec: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Updates helper arrays after candidates have been elected, transferring surplus votes.

        This method handles the vote transfer process when one or more candidates cross the
        election threshold. It moves ballot pointers to the next available candidate and
        adjusts vote weights according to the transfer method (fractional or random).

        Args:
            winners (list[int]): List of candidate indices who were elected this round.
            tallies (np.ndarray): Current vote tallies for all candidates.
            mutated_fpv_vec (np.ndarray): First preference vector (modified in place).
            mutated_wt_vec (np.ndarray): Weight vector for ballots (modified in place).
            bool_ballot_matrix (np.ndarray): Boolean mask indicating entries of the ballot matrix in eliminated_or_exhausted.
                (This has been already updated in find_winners.)
            mutated_pos_vec (np.ndarray): Position vector tracking current ballot positions.
            mutated_eliminated_or_exhausted (list[int]): List of all eliminated/elected candidates.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]: Updated helper arrays.
        """
        #Running example: say that self._ballot matrix looks like
        # [[2, 1, 3, -127],
        #  [0, 2, 1, -127],
        #  [2, 3, 1, -127],
        #  [2, -127, -127, -127]]

        # say that candidate 2 and 3 was already eliminated, and say that 1 just got elected.
        # in this case the fpv_vec looks like [1,0,1,-127] before update, and the bool_ballot_matrix looks like:
        # [[0, 1, 0, 1],
        #  [1, 0, 1, 1],
        #  [0, 0, 1, 1],
        #  [0, 1, 1, 1]]

        # rows_with_winner_fpv would look like [True, False, True, False] in the running example
        rows_with_winner_fpv = np.isin(mutated_fpv_vec, winners)

        # the winner_row_indices would look like [0, 2] in the running example, and the allowed_pos_matrix looks like
        # [[0, 1, 0, 1], #row 0 of the original matrix
        #  [0, 0, 1, 1]] #row 2 of the original matrix
        winner_row_indices = np.where(rows_with_winner_fpv)[0]
        allowed_pos_matrix = bool_ballot_matrix[winner_row_indices]
        
        #this tells us which index the following pos_vec will highlight -- in our example, [3, 3]
        next_fpv_pos_vec = allowed_pos_matrix.argmax(axis=1)
        mutated_pos_vec[winner_row_indices] = next_fpv_pos_vec

        # this tells us who the new fpv vote will be with the updated positions
        next_fpv_vec = self._ballot_matrix[winner_row_indices, next_fpv_pos_vec]
        # don't update fpv_vec yet because we need to know current fpv to transfer weights

        if self.transfer == "fractional":
            get_transfer_value = np.frompyfunc(
                lambda w: (tallies[w] - self.threshold) / tallies[w], 1, 1
            )
            transfer_value_values = get_transfer_value(mutated_fpv_vec[rows_with_winner_fpv]).astype(np.float64)
            mutated_wt_vec[rows_with_winner_fpv] *= transfer_value_values
            mutated_fpv_vec[rows_with_winner_fpv] = self._ballot_matrix[winner_row_indices, next_fpv_pos_vec]
        elif (
            self.transfer is not None and "random" in self.transfer 
        ):  
            new_weights = np.zeros_like(mutated_wt_vec, dtype=np.int64)
            for w in winners:
                if self.transfer == "cambridge_random":
                    # pre-emptively exhaust ballots that will be exhausted -- this is what Cambridge does
                    mutated_fpv_vec[winner_row_indices[next_fpv_vec == -127]] = -127
                surplus = int(tallies[w] - self.threshold)
                counts = self._sample_to_transfer(
                    fpv_vec=mutated_fpv_vec,
                    wt_vec=mutated_wt_vec,
                    winner=w,
                    surplus=surplus,
                    rng=None,
                )
                new_weights += counts.astype(new_weights.dtype)

            # set the new weights for rows in play to exactly the transferred amount
            mutated_wt_vec[winner_row_indices] = new_weights[winner_row_indices]
            mutated_fpv_vec[rows_with_winner_fpv] = next_fpv_vec
        return (
            mutated_fpv_vec,
            mutated_wt_vec,
            bool_ballot_matrix,
            mutated_pos_vec,
        )

    def __update_because_loser(
        self,
        loser: int,
        mutated_fpv_vec: np.ndarray,
        wt_vec: np.ndarray,
        bool_ballot_matrix: np.ndarray,
        mutated_pos_vec: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Updates helper arrays after a single candidate has been eliminated, transferring surplus votes.

        There's not a lot to do here -- find_loser already updates the mutant stencil,
        so we just need to move the pos and fpv vecs to their pre-calculated new positions.

        Args:
            loser (int): Index of the candidate who was eliminated this round.
            tallies (np.ndarray): Current vote tallies for all candidates.
            mutated_fpv_vec (np.ndarray): First preference vector (modified in place).
            mutated_wt_vec (np.ndarray): Weight vector for ballots (modified in place).
            bool_ballot_matrix (np.ndarray): Boolean mask indicating entries of the ballot matrix in eliminated_or_exhausted.
                (This has been already updated in find_loser).
            mutated_pos_vec (np.ndarray): Position vector tracking current ballot positions.
            mutated_eliminated_or_exhausted (list[int]): List of all eliminated/elected candidates.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]: Updated helper arrays.
        """

        # cf. update_because_winner for detailed example
        rows_with_loser_fpv = np.isin(mutated_fpv_vec, loser)
        loser_row_indices = np.where(rows_with_loser_fpv)[0]
        allowed_pos_matrix = bool_ballot_matrix[loser_row_indices]
        mutated_pos_vec[loser_row_indices]  = allowed_pos_matrix.argmax(axis=1)
        mutated_fpv_vec[loser_row_indices] = self._ballot_matrix[loser_row_indices, mutated_pos_vec[loser_row_indices]]

        return (
            mutated_fpv_vec,
            wt_vec,
            bool_ballot_matrix,
            mutated_pos_vec,
        )

    def __find_loser(
        self,
        tallies: np.ndarray,
        turn: int,
        mutant_bool_ballot_matrix: np.ndarray,
        mutant_winner_list: list[int],
        mutant_eliminated_or_exhausted: list[int],
        mutant_tiebreak_record: dict[int, tuple[list[int], int, int]],
    ) -> tuple[
        int,
        tuple[np.ndarray, list[int], list[int], dict[int, tuple[list[int], int, int]]],
    ]:
        """
        Identify the candidate to eliminate in the current round, applying tiebreaks if necessary.

        Args:
            tallies (np.ndarray): Current tallies for each candidate.
            initial_tally (np.ndarray): Initial tallies for tiebreaking.
            turn (int): The current round number.
            mutant_bool_ballot_matrix (np.ndarray): Boolean mask for eliminated candidates.
            mutant_winner_list (list[int]): List of winner candidate indices so far.
            mutant_eliminated_or_exhausted (list[int]): List of eliminated candidate indices so far.
            mutant_tiebreak_record (dict[int, tuple[list[int], int, int]]): Tiebreak record for each round.

        Returns:
            tuple: (index of eliminated candidate, updated state tuple)
        """
        masked_tallies = np.where(
            np.isin(np.arange(len(tallies)), mutant_eliminated_or_exhausted), np.inf, tallies
        ) # we must do this because sometimes there are cands with 0 FPV who have yet to be eliminated
        if np.count_nonzero(masked_tallies == np.min(masked_tallies)) > 1: #must run a tiebreak
            potential_losers: list[int] = np.where(
                masked_tallies == masked_tallies.min()
            )[0].astype(int).tolist()
            L, mutant_tiebreak_record = self.__run_loser_tiebreak(
                potential_losers, turn, mutant_tiebreak_record
            )
        else:
            L = int(np.argmin(masked_tallies))
        mutant_eliminated_or_exhausted.append(L)
        self.__update_bool_ballot_matrix(mutant_bool_ballot_matrix, [L])
        return L, (
            mutant_bool_ballot_matrix,
            mutant_winner_list,
            mutant_eliminated_or_exhausted,
            mutant_tiebreak_record,
        )

    def __find_winners(
        self,
        tallies: np.ndarray,
        turn: int,
        mutant_bool_ballot_matrix: np.ndarray,
        mutant_winner_list: list[int],
        mutant_eliminated_or_exhausted: list[int],
        mutant_tiebreak_record: dict[int, tuple[list[int], int, int]],
    ) -> tuple[
        list[int],
        tuple[np.ndarray, list[int], list[int], dict[int, tuple[list[int], int, int]]],
    ]:
        """
        Identify the candidate(s) to elect in the current round, applying tiebreaks if necessary.

        Args:
            tallies (np.ndarray): Current tallies for each candidate.
            turn (int): The current round number.
            mutant_bool_ballot_matrix (np.ndarray): Boolean mask for eliminated/elected candidates.
            mutant_winner_list (list[int]): List of winner candidate indices so far.
            mutant_eliminated_or_exhausted (list[int]): List of eliminated/elected candidate indices so far.
            mutant_tiebreak_record (dict[int, tuple[list[int], int, int]]): Tiebreak record for each round.

        Returns:
            tuple: (list of elected candidate indices, updated state tuple)
        """
        if self.simultaneous:
            winners = np.where(tallies >= self.threshold)[0]
            winners = winners[np.argsort(-tallies[winners])]
            winners = winners.tolist()
        else:
            if np.count_nonzero(tallies == np.max(tallies)) > 1:
                potential_winners: list[int] = np.where(
                    tallies == tallies.max()
                )[0].astype(int).tolist()
                w, mutant_tiebreak_record = self.__run_winner_tiebreak(
                    potential_winners, turn, mutant_tiebreak_record
                )
            else:
                w = int(np.argmax(tallies))
            winners = [w]
        for w in winners:
            mutant_winner_list.append(int(w))
            mutant_eliminated_or_exhausted.append(w)
        self.__update_bool_ballot_matrix(mutant_bool_ballot_matrix, winners)
        return winners, (
            mutant_bool_ballot_matrix,
            mutant_winner_list,
            mutant_eliminated_or_exhausted,
            mutant_tiebreak_record,
        )

    def __run_winner_tiebreak(self, tied_winners, turn, mutant_tiebreak_record):
        """
        Handle new winner tiebreaking logic.

        Args:
            tied_winners (list[int]): List of candidate indices that are tied.
            turn (int): The current round number.
            mutant_tiebreak_record (dict[int, tuple[list[int], int, int]]): Tiebreak record for each round.

        Returns:
            tuple: (index of new winner, updated tiebreak record)
        """
        packaged_tie = frozenset([self.candidates[w] for w in tied_winners])
        if self._winner_tiebreak == 'first_place':
            W, packaged_ranking = self.__fpv_tiebreak(tied_winners, 'winner')
        elif self._winner_tiebreak is not None:
           packaged_ranking = tiebreak_set(r_set=packaged_tie, profile=self.profile, tiebreak=self._winner_tiebreak)
           W = self.candidates.index(list(packaged_ranking[0])[0])
        else:
            raise ValueError(
                "Cannot elect correct number of candidates without breaking ties."
            )
        mutant_tiebreak_record[turn] = {packaged_tie: packaged_ranking}
        return W, mutant_tiebreak_record

    def __run_loser_tiebreak(self, tied_losers, turn, mutant_tiebreak_record):
        """
        Handle new loser tiebreaking logic.

        Args:
            tied_losers (list[int]): List of candidate indices that are tied.
            turn (int): The current round number.
            mutant_tiebreak_record (dict[int, tuple[list[int], int, int]]): Tiebreak record for each round.

        Returns:
            tuple: (index of new loser, updated tiebreak record)
        """
        packaged_tie = frozenset([self.candidates[w] for w in tied_losers])
        if self._loser_tiebreak == 'first_place':
            L, packaged_ranking = self.__fpv_tiebreak(tied_losers, 'loser')
        else:
            packaged_ranking = tiebreak_set(r_set=packaged_tie, profile=self.profile, tiebreak=self._loser_tiebreak)
            L = self.candidates.index(list(packaged_ranking[-1])[0]) #I hecking love fsets
        mutant_tiebreak_record[turn] = {packaged_tie: packaged_ranking}
        return L, mutant_tiebreak_record

    def __update_bool_ballot_matrix(
        self, _mutant_bool_ballot_matrix: np.ndarray, newly_gone: list[int]
    ) -> np.ndarray:
        """
        Update the stencil mask to mark candidates as eliminated or elected.

        Args:
            _mutant_bool_ballot_matrix (np.ndarray): Boolean mask of eliminated/elected candidates.
            newly_gone (list[int]): List of candidate indices to mark as eliminated/elected.

        Returns:
            np.ndarray: Updated stencil mask.
        """
        _mutant_bool_ballot_matrix &= ~np.isin(self._ballot_matrix, newly_gone)
        return _mutant_bool_ballot_matrix

    def _make_initial_fpv(self):
        return np.bincount(
            self._fpv_vec[self._fpv_vec != -127],
            weights=self._wt_vec[self._fpv_vec != -127],
            minlength=len(self.candidates),
        )

    def _run_STV(
        self,
        ballot_matrix: np.ndarray,
        wt_vec: np.ndarray,
        fpv_vec: np.ndarray,
        m: int,
        ncands: int,
    ) -> tuple[
        list[np.ndarray],
        list[tuple[int, list[int], np.ndarray, str]],
        dict[int, tuple[list[int], int, int]],
    ]:
        """
        This runs the STV algorithm.

        Args:
            ballot_matrix (np.ndarray[np.int8]): Matrix where each row is a ballot, each column is a ranking.
            wt_vec (np.ndarray[np.float64]): Each entry is the weight of the corresponding row in the ballot matrix.
                This vector is modified in place.
            fpv_vec (np.ndarray[np.int8]): Each entry is the first preference vote of the corresponding row in the ballot matrix.
                This vector is modified in place.
            m (int): The number of seats in the election.
            ncands (int): The number of candidates in the election.

        Returns:
            tuple[list[np.ndarray], list[tuple[int, list[int], np.ndarray, str]], dict[int, tuple[list[int], int, int]]]:
                The tally record is a list with one array per round;
                    each array counts the first-place votes for the remaining candidates.
                The play-by-play logs some information for the public methods:
                    - turn number
                    - list of candidates elected or eliminated this turn
                    - weight vector at this turn, if the turn was an election
                    - turn type: 'election', 'elimination', or 'default'
                The tiebreak record is a dictionary mapping turn number to a tuple of
                    (potential candidates involved in tiebreak, chosen candidate, tiebreak type).
        """
        fpv_by_round = []
        play_by_play: list[tuple[int, list[int], np.ndarray, str]] = []
        turn = 0
        quota = self.threshold
        winner_list: list[int] = []
        eliminated_or_exhausted: list[int] = []
        tiebreak_record: dict[int, tuple[list[int], int, int]] = dict()
        pos_vec = np.zeros(ballot_matrix.shape[0], dtype=np.int8)
        # this contains 1s in positions where the candidate has not been eliminated/exhausted
        mutant_bool_ballot_matrix = np.ones_like(ballot_matrix, dtype=bool)

        def make_tallies(fpv_vec, wt_vec, ncands):
            return np.bincount(
                fpv_vec[fpv_vec != -127],
                weights=wt_vec[fpv_vec != -127],
                minlength=ncands,
            )

        mutant_engine = (fpv_vec, wt_vec, mutant_bool_ballot_matrix, pos_vec)
        mutant_record = (mutant_bool_ballot_matrix, winner_list, eliminated_or_exhausted, tiebreak_record)
        # below is the main loop of the algorithm
        while len(winner_list) < m:
            # force the bincount to count entries 0 through ncands-1, even if some candidates have no votes
            tallies = make_tallies(fpv_vec, wt_vec, ncands)
            fpv_by_round.append(tallies.copy())
            while np.any(tallies >= quota):
                winners, mutant_record = self.__find_winners(
                    tallies, turn, *mutant_record
                )
                mutant_engine = self.__update_because_winner(
                    winners, tallies, *mutant_engine
                )
                play_by_play.append((turn, winners, np.array(wt_vec), "election"))
                turn += 1
                tallies = make_tallies(fpv_vec, wt_vec, ncands)
                fpv_by_round.append(tallies.copy())
            if len(winner_list) == m:
                break
            if len(eliminated_or_exhausted) - len(winner_list) == ncands - m:
                still_standing = [i for i in range(ncands) if i not in eliminated_or_exhausted]
                winner_list += still_standing
                play_by_play.append((turn, still_standing, np.array([]), "default"))
                turn += 1
                fpv_by_round.append(
                    np.zeros(ncands, dtype=np.float64)
                )  # this is needed for get_remaining to behave nicely
                break
            L, mutant_record = self.__find_loser(
                tallies, turn, *mutant_record
            )
            mutant_engine = self.__update_because_loser(L, *mutant_engine)
            play_by_play.append((turn, [L], np.array([]), "elimination"))
            turn += 1
        return fpv_by_round, play_by_play, tiebreak_record

    def get_remaining(self, round_number: int = -1) -> tuple[frozenset]:
        """
        Fetch the remaining candidates after the given round.

        Args:
            round_number (int): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            tuple[frozenset[str], ...]:
                Tuple of sets of remaining candidates. Ordering of tuple
                denotes ranking of remaining candidates, sets denote ties.
        """
        tallies = self._fpv_by_round[round_number]
        tallies_to_cands = dict()
        tallies_to_cands = {tally: [cand_string for c, cand_string in self.candidates.items() if tallies[c] == tally] for tally in tallies}
        tallies_to_cands = dict(
            sorted(
                tallies_to_cands.items(),
                key=lambda item: item[0],
                reverse=True,
            )
        )
        return (
            tuple(frozenset(value) for value in tallies_to_cands.values())
            if len(tallies_to_cands) > 0
            else (frozenset(),)
        )

    def get_elected(self, round_number: int = -1) -> tuple[frozenset[str], ...]:
        """
        Fetch the elected candidates up to the given round number.

        Args:
            round_number (int): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            tuple[frozenset[str], ...]:
                Tuple of winning candidates in order of election. Candidates
                in the same set were elected simultaneously, i.e. in the final ranking
                they are tied.
        """
        if (
            round_number < -len(self._fpv_by_round)
            or round_number > len(self._fpv_by_round) - 1
        ):
            raise IndexError("round_number out of range.")
        round_number = round_number % len(self._fpv_by_round)
        list_of_winners = [
            [c]
            for _, cand_list, _, turn_type in self._play_by_play[:round_number]
            if turn_type == "election"
            for c in cand_list
        ] + [
            cand_list
            for _, cand_list, _, turn_type in self._play_by_play[:round_number]
            if turn_type == "default"
        ]
        return tuple(
            frozenset([self.candidates[c] for c in w_list])
            for w_list in list_of_winners
        )

    def get_eliminated(self, round_number: int = -1) -> tuple[frozenset[str], ...]:
        """
        Fetch the eliminated candidates up to the given round number.

        Args:
            round_number (int): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            tuple[frozenset[str], ...]:
                Tuple of eliminated candidates in reverse order of elimination.
                Candidates in the same set were eliminated simultaneously, i.e. in the final ranking
                they are tied.
        """
        if (
            round_number < -len(self._fpv_by_round)
            or round_number > len(self._fpv_by_round) - 1
        ):
            raise IndexError("round_number out of range.")
        round_number = round_number % len(self._fpv_by_round)
        if round_number == 0:
            return tuple()
        list_of_losers = [
            cand_list
            for _, cand_list, _, turn_type in self._play_by_play[round_number - 1 :: -1]
            if turn_type == "elimination"
        ]
        return tuple(
            frozenset([self.candidates[c] for c in l_list]) for l_list in list_of_losers
        )

    def get_ranking(self, round_number: int = -1) -> tuple[frozenset[str], ...]:
        """
        Fetch the ranking of candidates after a given round.

        Args:
            round_number (int): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            tuple[frozenset[str],...]: Ranking of candidates.
        """
        # len condition handles empty remaining candidates
        return tuple(
            [
                s
                for s in self.get_elected(round_number)
                + self.get_remaining(round_number)
                + self.get_eliminated(round_number)
                if len(s) != 0
            ]
        )

    def get_status_df(self, round_number: int = -1) -> pd.DataFrame:
        """
        Yield the status (elected, eliminated, remaining) of the candidates in the given round.
        DataFrame is sorted by current ranking.

        Args:
            round_number (int): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            pd.DataFrame:
                Data frame displaying candidate, status (elected, eliminated,
                remaining), and the round their status updated.
        """
        status_df = pd.DataFrame(
            {
                "Status": ["Remaining"] * len(self.candidates),
                "Round": [0] * len(self.candidates),
            },
            index=self.candidates,
        )
        if (
            round_number < -len(self._fpv_by_round)
            or round_number > len(self._fpv_by_round) - 1
        ):
            raise IndexError("round_number out of range.")

        round_number = round_number % len(self._fpv_by_round)
        new_index = [c for s in self.get_ranking(round_number) for c in s]

        for turn_id, birthday_list, _, turn_type in self._play_by_play[:round_number]:
            if turn_type == "elimination":  # loser
                status_df.at[self.candidates[birthday_list[0]], "Status"] = "Eliminated"
                status_df.at[self.candidates[birthday_list[0]], "Round"] = turn_id + 1
            elif turn_type == "election":  # winner
                for c in birthday_list:
                    status_df.at[self.candidates[c], "Status"] = "Elected"
                    status_df.at[self.candidates[c], "Round"] = turn_id + 1
            elif turn_type == "default":  # winner by default
                for c in birthday_list:
                    status_df.at[self.candidates[c], "Status"] = "Elected"
                    status_df.at[self.candidates[c], "Round"] = turn_id + 1
        # iterating through the rows of status_df, change "Round" to round_number if "Status" is still "Remaining"
        for c in self.candidates:
            if status_df.at[c, "Status"] == "Remaining":
                status_df.at[c, "Round"] = round_number
        status_df = status_df.reindex(new_index)
        return status_df

    def _make_election_states(self):
        e_states = [
            ElectionState(
                round_number=0,
                remaining=self.get_remaining(0),
                scores={
                    self.candidates[c]: self._fpv_by_round[0][c]
                    for c in self._fpv_by_round[0].nonzero()[0]
                },
            )
        ]
        for i, play in enumerate(self._play_by_play):
            packaged_tiebreak = self._tiebreak_record.get(i, dict())
            packaged_elected = tuple([frozenset([self.candidates[c]]) for c in play[1]]
                            ) if play[-1] != "elimination" else (frozenset(),)
            packaged_eliminated=(frozenset([self.candidates[c] for c in play[1]]),
                        ) if play[-1] == "elimination" else (frozenset(),)
            packaged_scores={
                self.candidates[c]: self._fpv_by_round[i + 1][c]
                for c in self._fpv_by_round[i + 1].nonzero()[0]
            }
            e_states.append(
                    ElectionState(
                        round_number=i + 1,
                        remaining=self.get_remaining(i + 1),
                        elected=packaged_elected,
                        tiebreaks=packaged_tiebreak,
                        eliminated=packaged_eliminated,
                        scores=packaged_scores,
                    )
                )

        return e_states

    def get_profile(self, round_number: int = -1) -> PreferenceProfile:
        """
        Fetch the PreferenceProfile of the given round number.

        Args:
            round_number (int): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            PreferenceProfile

        """
        if (
            round_number < -len(self.election_states)
            or round_number > len(self.election_states) - 1
        ):
            raise IndexError("round_number out of range.")

        round_number = round_number % len(self.election_states)

        remaining = self._fpv_by_round[round_number].nonzero()[0]
        wt_vec = self._wt_vec.copy()
        if self.m > 1: # this loop could be avoided if we improved how wt_vec is recorded, but the loop is also quite small
            for i in range(len(self._play_by_play[:round_number]) - 1, -1, -1):
                if self._play_by_play[i][-1] == "election":
                    wt_vec = self._play_by_play[i][2]
                    break

        idx_to_fset = {c: frozenset([self.candidates[c]]) for c in remaining}

        # --- 1) drop last column by view ---
        A = self._ballot_matrix.copy()
        A = A[:, :-1]

        n_rows, n_cols = A.shape

        # --- 2) keep only entries in `remaining` ---
        remaining_arr = np.fromiter((int(x) for x in remaining), dtype=np.int64)
        keep_mask = np.isin(A, remaining_arr)

        # --- 3) stable left-compaction, fill with -127 ---
        out = np.full_like(A, fill_value=-127)               # int8
        pos = keep_mask.cumsum(axis=1) - 1                       # target col for each kept entry
        r_idx, c_idx = np.nonzero(keep_mask)
        out[r_idx, pos[r_idx, c_idx]] = A[r_idx, c_idx]

        # --- 4) int8 -> frozenset mapping via 256-entry LUT ---
        # default for anything missing (including -127): frozenset("~")
        lut: np.ndarray = np.empty(256, dtype=object)
        lut[:] = frozenset(["~"])
        for k, v in idx_to_fset.items():
            lut[int(np.int16(k)) + 128] = v
        # index into LUT (shift by +128 to map [-128,127] -> [0,255])
        obj = lut[out.astype(np.int16) + 128]                    # dtype=object, frozensets

        # --- 5) to DataFrame with Ranking_i columns ---
        data = {f"Ranking_{i+1}": obj[:, i] for i in range(n_cols)}
        df = pd.DataFrame(data)

        # --- 6) Ballot Index column & set as index ---
        df.insert(0, "Ballot Index", np.arange(n_rows, dtype=int))
        df.set_index("Ballot Index", inplace=True)

        # --- 7) Voter Set: empty set per row (distinct objects) ---
        df["Voter Set"] = [set() for _ in range(n_rows)]

        # --- 8) Weight column ---
        df["Weight"] = wt_vec.astype(np.float64, copy=False)

        return condense_profile(PreferenceProfile(contains_rankings=True, max_ranking_length=self.profile.max_ranking_length, candidates=tuple([self.candidates[c] for c in remaining]), df=df))

    def get_step(
        self, round_number: int = -1
    ) -> tuple[PreferenceProfile, ElectionState]:
        """
        Fetches the profile and ElectionState of the given round number.

        Args:
            round_number (int): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            tuple[PreferenceProfile, ElectionState]
        """
        return (self.get_profile(round_number), self.election_states[round_number])

    def _sample_to_transfer(
        self, fpv_vec: np.ndarray, wt_vec: np.ndarray, winner: int, surplus: int, rng=None
    ) -> np.ndarray:
        """
        Samples s row indices to transfer from an implicit pool,
        where each row index i appears wt_vec[i] times if fpv_vec[i] == winner.
        Returns a counts vector where counts[i] is the number of times row i was sampled.
        Ensures sum(counts) == s and counts[i] <= wt_vec[i].

        Args:
            fpv_vec (np.ndarray[np.int8]): First-preference vector.
            wt_vec (np.ndarray[np.float64]): Weights vector.
            w (int): Candidate code whose ballots are to be transferred.
            s (int): Number of surplus votes to transfer.
            rng (np.random.Generator, optional): Random number generator. If None, a new default
                generator is created. Defaults to None.
        """
        if rng is None:
            rng = np.random.default_rng()

        #running example: assume that candidate 2 just won.
        #assume the fpv_vec looks like [2,5,3,2]
        #then eligible looks like [True, False, False, True]
        #and winner_row_indices looks like [0, 3]
        eligible = fpv_vec == winner
        winner_row_indices = np.flatnonzero(eligible)

        # assume the original weight vector was [200, 100, 50, 25]
        # then wts looks like [200, 25]
        wts = wt_vec[winner_row_indices].astype(np.int64)

        # assume that quota was 220, so winner 2 had 5 surplus votes and 225 transferable votes
        transferable = int(wts.sum())

        # this deals with cases where there are fewer than surplus votes to transfer (lots of exhausted ballots)
        surplus = min(surplus, transferable)

        # Sample surplus distinct positions in the implicit pool [0, transferable)
        # in our example: we sample 5 distinct numbers from [0, 225)
        positions_to_transfer = rng.choice(transferable, size=surplus, replace=False)
        positions_to_transfer.sort()

        # Say we sampled the numbers 12, 50, 178, 200, and 201
        # numbers 0 through 199 inclusive get mapped to the first bin, so the first three sampled votes go to winner_row_index[0]
        # numbers 200 and 201 get mapped to the second bin, so they go to our second winner_row_index[1]
        bins = np.cumsum(wts)  # len = len(idx)
        owners = np.searchsorted(bins, positions_to_transfer, side="right")  # values in winner_row_indices

        # Accumulate counts back to global rows
        counts_local = np.bincount(
            owners, minlength=winner_row_indices.size
        )
        counts = np.zeros(fpv_vec.shape[0], dtype=np.int64) 
        counts[winner_row_indices] = counts_local #this tells us how many times each row was sampled as indexed in the global ballot_matrix
        return counts

    def __str__(self):
        return self.get_status_df().to_string(index=True, justify="justify")

    __repr__ = __str__


class STV(RankingElection):
    """
    STV elections. All ballots must have no ties.

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        m (int): Number of seats to be elected. Defaults to 1.
        transfer (Callable[[str, float, Union[tuple[Ballot], list[Ballot]], int], tuple[Ballot,...]]):
        Transfer method. Defaults to fractional transfer.
            Function signature is elected candidate, their number of first-place votes, the list of
            ballots with them ranked first, and the threshold value. Returns the list of ballots
            after transfer.
        quota (str): Formula to calculate quota. Accepts "droop" or "hare".
            Defaults to "droop".
        simultaneous (bool): True if all candidates who cross threshold in a round are
            elected simultaneously, False if only the candidate with highest first-place votes
            who crosses the threshold is elected in a round. Defaults to False.
        tiebreak (str): Method to be used if a tiebreak is needed. Accepts
            'borda' and 'random'. Defaults to None, in which case a ValueError is raised if
            a tiebreak is needed.

    """

    def __init__(
        self,
        profile: PreferenceProfile,
        m: int = 1,
        transfer: Callable[
            [str, float, Union[tuple[Ballot], list[Ballot]], int],
            tuple[Ballot, ...],
        ] = fractional_transfer,
        quota: str = "droop",
        simultaneous: bool = True,
        tiebreak: Optional[str] = None,
    ):
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

    def _stv_validate_profile(self, profile: PreferenceProfile):
        """
        Validate that each ballot has a ranking, and that there are no ties in ballots.
        """
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
                    f"Ballot {Ballot(ranking=tuple(row.tolist()), weight = weight_col[idx])} "
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
        elected_cands = [c for s in self.get_elected() for c in s]

        if len(elected_cands) == self.m:
            return True
        return False

    def _simultaneous_elect_step(
        self, profile: PreferenceProfile, prev_state: ElectionState
    ) -> tuple[tuple[frozenset[str], ...], PreferenceProfile]:
        """
        Run one step of an election from the given profile and previous state.
        Used for simultaneous STV election if candidates cross threshold.

        Args:
            profile (PreferenceProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.

        Returns:
            tuple[tuple[frozenset[str],...], PreferenceProfile]:
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
        new_ballots = [Ballot()] * profile.num_ballots
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
            condense_ballot_ranking(
                remove_cand_from_ballot([c for s in elected for c in s], b)
            )
            for b in new_ballots
            if b.ranking
        )

        remaining_cands = set(profile.candidates_cast).difference(
            [c for s in elected for c in s]
        )

        new_profile = PreferenceProfile(
            ballots=cleaned_ballots,
            candidates=tuple(remaining_cands),
            max_ranking_length=profile.max_ranking_length,
        )
        return (tuple(elected), new_profile)

    def _single_elect_step(
        self, profile: PreferenceProfile, prev_state: ElectionState
    ) -> tuple[
        tuple[frozenset[str], ...],
        dict[frozenset[str], tuple[frozenset[str], ...]],
        PreferenceProfile,
    ]:
        """
        Run one step of an election from the given profile and previous state.
        Used for one-by-one STV election if candidates cross threshold.

        Args:
            profile (PreferenceProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.

        Returns:
            tuple[tuple[frozenset[str],...], dict[frozenset[str], tuple[frozenset[str],...]],
            PreferenceProfile]:
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
        new_ballots = [Ballot()] * profile.num_ballots
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
            condense_ballot_ranking(remove_cand_from_ballot(elected_c, b))
            for b in new_ballots
            if b.ranking
        )

        remaining_cands = set(profile.candidates_cast).difference(
            [c for s in elected for c in s]
        )
        new_profile = PreferenceProfile(
            ballots=cleaned_ballots,
            candidates=tuple(remaining_cands),
            max_ranking_length=profile.max_ranking_length,
        )
        return elected, tiebreaks, new_profile

    def _run_step(
        self, profile: PreferenceProfile, prev_state: ElectionState, store_states=False
    ) -> PreferenceProfile:
        """
        Run one step of an election from the given profile and previous state.
        STV sets a threshold for first-place votes. If a candidate passes it, they are elected.
        We remove them from all ballots and transfer any surplus ballots to other candidates.
        If no one passes, we eliminate the lowest ranked candidate and reallocate their ballots.

        Can be run 1-by-1 or simultaneous, which determines what happens if multiple people cross
        threshold.

        Args:
            profile (PreferenceProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.
            store_states (bool, optional): True if `self.election_states` should be updated with the
                ElectionState generated by this round. This should only be True when used by
                `self._run_election()`. Defaults to False.

        Returns:
            PreferenceProfile: The profile of ballots after the round is completed.
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
            new_profile = PreferenceProfile()

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

            new_profile = remove_and_condense_ranked_profile(
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
    IRV (Instant-runoff voting) elections.  Elect 1 seat. All ballots must have no ties.
    Equivalent to STV for m = 1.

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        quota (str, optional): Formula to calculate quota. Accepts "droop" or "hare".
            Defaults to "droop".
        tiebreak (str, optional): Method to be used if a tiebreak is needed. Accepts
            'borda' and 'random'. Defaults to None, in which case a ValueError is raised if
            a tiebreak is needed.

    """

    def __init__(
        self,
        profile: PreferenceProfile,
        quota: str = "droop",
        tiebreak: Optional[str] = None,
    ):
        super().__init__(profile, m=1, quota=quota, tiebreak=tiebreak)


class SequentialRCV(STV):
    """
    An STV election in which votes are not transferred after a candidate has reached threshold, or
    been elected. This system is actually used in parts of Utah.

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        m (int, optional): Number of seats to be elected. Defaults to 1.
        quota (str, optional): Formula to calculate quota. Accepts "droop" or "hare".
            Defaults to "droop".
        simultaneous (bool, optional): True if all candidates who cross threshold in a round are
            elected simultaneously, False if only the candidate with highest first-place votes
            who crosses the threshold is elected in a round. Defaults to True.
        tiebreak (str, optional): Method to be used if a tiebreak is needed. Accepts
            'borda' and 'random'. Defaults to None, in which case a ValueError is raised if
            a tiebreak is needed.

    """

    def __init__(
        self,
        profile: PreferenceProfile,
        m: int = 1,
        quota: str = "droop",
        simultaneous: bool = True,
        tiebreak: Optional[str] = None,
    ):
        def _transfer(
            winner: str,
            _fpv: float,
            ballots: Union[tuple[Ballot], list[Ballot]],
            _threshold: int,
        ) -> tuple[Ballot, ...]:
            del _fpv, _threshold  # unused and del on atomics is okay
            return tuple(
                condense_ballot_ranking(remove_cand_from_ballot(winner, b))
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
