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
)
from typing import Optional, Callable, Union
import pandas as pd
import numpy as np


class fast_STV:
    """
    STV elections. All ballots must have no ties.

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        m (int): Number of seats to be elected. Defaults to 1.
        transfer (str): Transfer method to be used. Accepts 'fractional' and 'random'. Defaults to 'fractional'.
        quota (str): Formula to calculate quota. Accepts "droop" or "hare". Defaults to "droop".
            Defaults to "droop".
        simultaneous (bool): True if all candidates who cross threshold in a round are
            elected simultaneously, False if only the candidate with highest first-place votes
            who crosses the threshold is elected in a round. Defaults to True.
        tiebreak (Optional[str]): Method to be used if a tiebreak is needed. Accepts
            'borda' and 'random'. Defaults to None, in which case a ValueError is raised if
            a tiebreak is needed.

    """

    def __init__(
        self,
        profile: PreferenceProfile,
        m: int = 1,
        transfer: Optional[str] = "fractional",
        quota: str = "droop",
        simultaneous: bool = False,
        tiebreak: Optional[str] = None,
    ):
        self._stv_validate_profile(profile)

        if m <= 0:
            raise ValueError("m must be positive.")
        elif len(profile.candidates_cast) < m:
            raise ValueError("Not enough candidates received votes to be elected.")
        self.m = m
        if transfer not in ["fractional", "random"]:
            raise ValueError("Transfer method must be either 'fractional' or 'random'.")
        self.transfer = transfer
        self.quota = quota

        self._ballot_length = profile.max_ranking_length

        self.threshold = self._get_threshold(profile.total_ballot_wt)
        self.simultaneous = simultaneous
        self.tiebreak = tiebreak

        self.candidates = list(profile.candidates)

        self._ballot_matrix, self._wt_vec, self._fpv_vec = self._convert_df(profile)
        self._winners, self._tally_record, self._play_by_play, self._tiebreak_record = (
            self._run_STV(
                self._ballot_matrix,
                np.array(self._wt_vec),
                self._fpv_vec,
                m,
                len(self.candidates),
            )
        )
        self.election_states = self._election_states()

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
                    f"Ballot {Ballot(ranking=tuple(row.to_list()), weight = weight_col[idx])} "
                    "contains a tied ranking."
                )
            if (row == tilde).all():
                raise TypeError("Ballots must have rankings.")

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

    def _convert_df(self, profile: PreferenceProfile):
        """
        This converts the profile into a numpy matrix with some helper arrays for faster iteration.
        """
        df = profile.df
        candidate_to_index = {
            name: i for i, name in enumerate(self.candidates)
        }  # canonical ordering or riot
        ranking_columns = sorted(
            [col for col in df.columns if col.startswith("Ranking")]
        )

        num_rows = len(df)
        num_cols = len(ranking_columns)

        # +1 column for padding -- adds some data overhead, but this might not be avoidable? you've gotta communicate ballot length somehow
        ballot_matrix = np.full(
            (num_rows, num_cols + 1), fill_value=-127, dtype=np.int8
        )
        wt_vec = np.empty(num_rows, dtype=np.float64)

        for col_idx, col in enumerate(ranking_columns):
            col_values = df[col].to_numpy()
            for row_idx, frozenset_entry in enumerate(col_values):
                val = next(
                    iter(frozenset_entry)
                ).strip()  # unwrap the frozenset; I hecking love frozensets
                if val == "~":
                    ballot_matrix[row_idx, col_idx] = (
                        -127
                    )  # possible TODO: improve this iteration not to look at ballot entries beyond the first empty one -- maybe by removing rows from the df as we go?
                else:
                    try:
                        ballot_matrix[row_idx, col_idx] = candidate_to_index[val]
                    except KeyError:
                        raise ValueError(f"Candidate '{val}' not in candidate list.")

        wt_vec[:] = df["Weight"].astype(np.float64).to_numpy()
        fpv_vec = ballot_matrix[:, 0].copy()

        return ballot_matrix, wt_vec, fpv_vec

    def _run_STV(
        self, ballot_matrix, wt_vec, fpv_vec, m, ncands
    ) -> tuple[
        list[int], list[np.ndarray], list[tuple[int, list[int], np.ndarray, int]], dict
    ]:
        """
        This runs the STV algorithm. Based.
        """
        tally_record = []
        play_by_play = []
        turn = 0
        quota = self.threshold
        winner_list = []
        gone_list = []
        tiebreak_record = dict()
        pos_vec = np.zeros(ballot_matrix.shape[0], dtype=np.int8)
        while len(winner_list) < m:
            # force the bincount to count entries 0 through ncands-1, even if some candidates have no votes
            tallies = np.bincount(
                fpv_vec[fpv_vec != -127],
                weights=wt_vec[fpv_vec != -127],
                minlength=ncands,
            )  # don't count -127 entries from fpv_vec
            tally_record.append(tallies.copy())
            while np.any(tallies >= quota):
                if self.simultaneous:
                    winners = np.where(tallies >= quota)[0]
                    # re-order winners by tally size
                    winners = winners[np.argsort(-tallies[winners])]
                    tau_values = {}
                    for w in winners:
                        winner_list.append(int(w))
                        gone_list.append(w)
                        tau_values[w] = (tallies[w] - quota) / tallies[w]
                    if self.transfer == "fractional":
                        for i in range(len(fpv_vec)):
                            if fpv_vec[i] in winners:
                                tau = tau_values[int(fpv_vec[i])]
                                while (
                                    ballot_matrix[i, pos_vec[i]] in gone_list
                                ):  # this must end by the time we reach the last column, which is padded with -127
                                    pos_vec[i] += 1
                                fpv_vec[i] = ballot_matrix[i, pos_vec[i]]
                                wt_vec[i] *= tau
                    elif self.transfer == "random":
                        new_weights = dict()
                        for w in winners:
                            transfer_bundle = self._sample_to_transfer(
                                fpv_vec, wt_vec, w, int(tallies[w] - quota)
                            )
                            new_weights[w] = np.bincount(
                                transfer_bundle, minlength=len(fpv_vec)
                            )
                        for i in range(len(fpv_vec)):
                            if fpv_vec[i] in winners:
                                while ballot_matrix[i, pos_vec[i]] in gone_list:
                                    pos_vec[i] += 1
                                fpv_vec[i] = ballot_matrix[i, pos_vec[i]]
                                wt_vec[i] = new_weights[fpv_vec[i]][i]
                    play_by_play.append((turn, winners, np.array(wt_vec), 1))
                    turn += 1
                else:
                    # check if tallies attains its maximum twice
                    if np.count_nonzero(tallies == np.max(tallies)) > 1:
                        potential_winners = np.where(tallies == np.max(tallies))[0]
                        if self.tiebreak is None:
                            raise ValueError(
                                "Cannot elect correct number of candidates without breaking ties."
                            )
                        if self.tiebreak == "random":
                            w = np.random.choice(potential_winners)
                            tiebreak_record[turn] = (potential_winners.tolist(), w, 1)
                        elif self.tiebreak == "borda":
                            borda_scores = np.zeros_like(
                                potential_winners, dtype=np.float64
                            )
                            for j in range(ballot_matrix.shape[0]):
                                for i in range(pos_vec[j], ballot_matrix.shape[1]):
                                    if ballot_matrix[j, i] in potential_winners:
                                        borda_scores[
                                            np.where(
                                                potential_winners == ballot_matrix[j, i]
                                            )[0][0]
                                        ] += wt_vec[j] * (
                                            ballot_matrix.shape[1] - i + pos_vec[j]
                                        )
                            w = potential_winners[
                                np.argmax(borda_scores)
                            ]  # it's possible the borda scores are tied too? oh well
                            tiebreak_record[turn] = (potential_winners.tolist(), w, 1)
                    else:
                        w = np.argmax(tallies)
                    winner_list.append(int(w))
                    gone_list.append(w)
                    tau = (tallies[w] - quota) / tallies[w]
                    if self.transfer == "fractional":
                        for i in range(len(fpv_vec)):
                            if fpv_vec[i] == w:
                                while (
                                    ballot_matrix[i, pos_vec[i]] in gone_list
                                ):  # again, we're padded
                                    pos_vec[i] += 1
                                fpv_vec[i] = ballot_matrix[i, pos_vec[i]]
                                wt_vec[i] *= tau
                    elif self.transfer == "random":
                        transfer_bundle = self._sample_to_transfer(
                            fpv_vec, wt_vec, w, int(tallies[w] - quota)
                        )
                        new_weights = np.bincount(
                            transfer_bundle, minlength=len(fpv_vec)
                        )
                        for i in range(len(fpv_vec)):
                            if fpv_vec[i] == w:
                                while ballot_matrix[i, pos_vec[i]] in gone_list:
                                    pos_vec[i] += 1
                                fpv_vec[i] = ballot_matrix[i, pos_vec[i]]
                                wt_vec[i] = new_weights[i]
                    play_by_play.append((turn, [w], np.array(wt_vec), 1))
                    turn += 1
                tallies = np.bincount(
                    fpv_vec[fpv_vec != -127],
                    weights=wt_vec[fpv_vec != -127],
                    minlength=ncands,
                )
                tally_record.append(tallies.copy())
            if len(winner_list) == m:
                return winner_list, tally_record, play_by_play, tiebreak_record
            if len(gone_list) - len(winner_list) == ncands - m:
                still_standing = [i for i in range(ncands) if i not in gone_list]
                winner_list += still_standing
                play_by_play.append((turn, still_standing, [], 2))
                turn += 1
                tally_record.append(
                    np.zeros(ncands, dtype=np.float64)
                )  # this is needed for get_remaining to behave nicely
                return winner_list, tally_record, play_by_play, tiebreak_record
            # masked tallies ignores indices in gone_list only (potentially leaving in candidates with 0 FPVs if they were not eliminated yet)
            masked_tallies = np.where(
                np.isin(np.arange(len(tallies)), gone_list), np.inf, tallies
            )  # used to be np.where(tallies > 0, tallies, np.inf)
            if (
                np.count_nonzero(masked_tallies == np.min(masked_tallies)) > 1
            ):  # do something funny if masked_tallies attains the minimum twice
                # count FPV votes of each I guess
                potential_losers = np.where(masked_tallies == np.min(masked_tallies))[0]
                L = potential_losers[
                    np.argmin(tally_record[0][potential_losers])
                ]  # it's possible this is tied too. oh well
                tiebreak_record[turn] = (potential_losers.tolist(), L, 0)
            else:
                L = np.argmin(masked_tallies)
            gone_list.append(L)
            for i in range(len(fpv_vec)):
                if fpv_vec[i] == L:
                    while ballot_matrix[i, pos_vec[i]] in gone_list:
                        pos_vec[i] += 1
                    fpv_vec[i] = ballot_matrix[i, pos_vec[i]]
            play_by_play.append((turn, [L], [], 0))
            turn += 1
        return winner_list, tally_record, play_by_play, tiebreak_record

    def get_remaining(self, round_number: int = -1) -> tuple[frozenset]:
        """
        Fetch the remaining candidates after the given round.

        Args:
            round_number (int): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            tuple[frozenset[str],...]:
                Tuple of sets of remaining candidates. Ordering of tuple
                denotes ranking of remaining candidates, sets denote ties.
        """
        tallies = self._tally_record[round_number]
        # weird dict structure to detect ties, which must be put in the same fset
        this_is_great_and_not_weird = dict()
        for c, t in enumerate(tallies):
            if t > 0:
                if t not in this_is_great_and_not_weird:
                    this_is_great_and_not_weird[t] = [self.candidates[c]]
                else:
                    this_is_great_and_not_weird[t].append(self.candidates[c])
        this_is_great_and_not_weird = dict(
            sorted(
                this_is_great_and_not_weird.items(),
                key=lambda item: item[0],
                reverse=True,
            )
        )
        return (
            tuple(frozenset(value) for value in this_is_great_and_not_weird.values())
            if len(this_is_great_and_not_weird) > 0
            else (frozenset(),)
        )

    def get_elected(self, round_number: int = -1) -> tuple[frozenset]:
        """
        Fetch the elected candidates up to the given round number.

        Args:
            round_number (int): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            tuple[frozenset[str],...]:
                List of winning candidates in order of election. Candidates
                in the same set were elected simultaneously, i.e. in the final ranking
                they are tied.
        """
        if (
            round_number < -len(self._tally_record)
            or round_number > len(self._tally_record) - 1
        ):
            raise IndexError("round_number out of range.")
        round_number = round_number % len(self._tally_record)
        list_of_winners = [
            [c]
            for _, cand_list, _, turn_type in self._play_by_play[:round_number]
            if turn_type == 1
            for c in cand_list
        ] + [
            cand_list
            for _, cand_list, _, turn_type in self._play_by_play[:round_number]
            if turn_type == 2
        ]
        return tuple(
            frozenset([self.candidates[c] for c in w_list])
            for w_list in list_of_winners
        )

    def get_eliminated(self, round_number: int = -1) -> tuple[frozenset]:
        """
        Fetch the eliminated candidates up to the given round number.

        Args:
            round_number (int): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            tuple[frozenset[str],...]:
                Tuple of eliminated candidates in reverse order of elimination.
                Candidates in the same set were eliminated simultaneously, i.e. in the final ranking
                they are tied.
        """
        if (
            round_number < -len(self._tally_record)
            or round_number > len(self._tally_record) - 1
        ):
            raise IndexError("round_number out of range.")
        round_number = round_number % len(self._tally_record)
        if round_number == 0:
            return tuple()
        list_of_losers = [
            cand_list
            for _, cand_list, _, turn_type in self._play_by_play[round_number - 1 :: -1]
            if turn_type == 0
        ]
        return tuple(
            frozenset([self.candidates[c] for c in l_list]) for l_list in list_of_losers
        )

    def get_ranking(self, round_number: int = -1) -> tuple[frozenset[str], ...]:
        """
        Fetch the ranking of candidates after a given round.

        Args:
            round_number (int, optional): The round number. Supports negative indexing. Defaults to
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
            round_number < -len(self._tally_record)
            or round_number > len(self._tally_record) - 1
        ):
            raise IndexError("round_number out of range.")

        round_number = round_number % len(self._tally_record)
        new_index = [c for s in self.get_ranking(round_number) for c in s]

        for turn_id, birthday_list, _, turn_type in self._play_by_play[:round_number]:
            if turn_type == 0:  # loser
                status_df.at[self.candidates[birthday_list[0]], "Status"] = "Eliminated"
                status_df.at[self.candidates[birthday_list[0]], "Round"] = turn_id + 1
            elif turn_type == 1:  # winner
                for c in birthday_list:
                    status_df.at[self.candidates[c], "Status"] = "Elected"
                    status_df.at[self.candidates[c], "Round"] = turn_id + 1
            elif turn_type == 2:  # winner by default
                for c in birthday_list:
                    status_df.at[self.candidates[c], "Status"] = "Elected"
                    status_df.at[self.candidates[c], "Round"] = turn_id + 1
        # iterating through the rows of status_df, change "Round" to round_number if "Status" is still "Remaining"
        for c in self.candidates:
            if status_df.at[c, "Status"] == "Remaining":
                status_df.at[c, "Round"] = round_number
        status_df = status_df.reindex(new_index)
        return status_df

    def _election_states(self):
        e_states = [
            ElectionState(
                round_number=0,
                remaining=self.get_remaining(0),
                scores={
                    self.candidates[c]: self._tally_record[0][c]
                    for c in self._tally_record[0].nonzero()[0]
                },
            )
        ]
        for i, play in enumerate(self._play_by_play):
            if i in self._tiebreak_record.keys():
                tiebreak = self._tiebreak_record[
                    i
                ]  # tiebreak_record[turn] = (potential_winners.tolist(), w, 1)
                w = tiebreak[1]
                tied_cands = tiebreak[0]
                denouement_list = [
                    self.candidates[c] for c in tied_cands if c != tiebreak[1]
                ]
                if tiebreak[-1] == 1:
                    denouement = (
                        frozenset([self.candidates[w]]),
                        frozenset([c for c in denouement_list]),
                    )
                if tiebreak[-1] == 0:
                    denouement = (
                        frozenset([c for c in denouement_list]),
                        frozenset([self.candidates[w]]),
                    )
                formatted_tiebreak = {
                    frozenset([self.candidates[c] for c in tied_cands]): denouement
                }
            else:
                formatted_tiebreak = None
            if play[-1] == 0:
                if formatted_tiebreak is None:
                    e_states.append(
                        ElectionState(
                            round_number=i + 1,
                            remaining=self.get_remaining(i + 1),
                            elected=(frozenset(),),
                            eliminated=(
                                frozenset([self.candidates[c] for c in play[1]]),
                            ),
                            scores={
                                self.candidates[c]: self._tally_record[i + 1][c]
                                for c in self._tally_record[i + 1].nonzero()[0]
                            },
                        )
                    )
                else:
                    e_states.append(
                        ElectionState(
                            round_number=i + 1,
                            remaining=self.get_remaining(i + 1),
                            elected=(frozenset(),),
                            tiebreaks=formatted_tiebreak,
                            eliminated=(
                                frozenset([self.candidates[c] for c in play[1]]),
                            ),
                            scores={
                                self.candidates[c]: self._tally_record[i + 1][c]
                                for c in self._tally_record[i + 1].nonzero()[0]
                            },
                        )
                    )
            elif play[-1] == 1:
                if formatted_tiebreak is None:
                    e_states.append(
                        ElectionState(
                            round_number=i + 1,
                            remaining=self.get_remaining(i + 1),
                            elected=tuple(
                                [frozenset([self.candidates[c]]) for c in play[1]]
                            ),
                            eliminated=(frozenset(),),
                            scores={
                                self.candidates[c]: self._tally_record[i + 1][c]
                                for c in self._tally_record[i + 1].nonzero()[0]
                            },
                        )
                    )
                else:
                    e_states.append(
                        ElectionState(
                            round_number=i + 1,
                            remaining=self.get_remaining(i + 1),
                            elected=tuple(
                                [frozenset([self.candidates[c]]) for c in play[1]]
                            ),
                            tiebreaks=formatted_tiebreak,
                            eliminated=(frozenset(),),
                            scores={
                                self.candidates[c]: self._tally_record[i + 1][c]
                                for c in self._tally_record[i + 1].nonzero()[0]
                            },
                        )
                    )
            elif play[-1] == 2:
                e_states.append(
                    ElectionState(
                        round_number=i + 1,
                        remaining=self.get_remaining(i + 1),
                        elected=tuple(
                            frozenset([self.candidates[c] for c in play[1]])
                            for c in play[1]
                        ),
                        eliminated=(frozenset(),),
                        scores={
                            self.candidates[c]: self._tally_record[i + 1][c]
                            for c in self._tally_record[i + 1].nonzero()[0]
                        },
                    )
                )

        return e_states

    def get_profile(self, round_number: int = -1) -> PreferenceProfile:
        """
        Fetch the PreferenceProfile of the given round number.

        Args:
            round_number (int, optional): The round number. Supports negative indexing. Defaults to
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

        hopeful = self._tally_record[round_number].nonzero()[0]
        ballots = []
        wt_vec = self._wt_vec.copy()
        if self.m > 1:
            # find the last entry in play_by_play[:round_number] with a 1 in the last position (there may be no such entry)
            for i in range(len(self._play_by_play[:round_number]) - 1, -1, -1):
                if self._play_by_play[i][-1] == 1:
                    wt_vec = self._play_by_play[i][2]
                    break
        # print(wt_vec)
        for i, row in enumerate(self._ballot_matrix):
            ballot = []
            for entry in row:
                if entry == -127:
                    break
                elif entry in hopeful:
                    ballot.append(frozenset([self.candidates[entry]]))
            if len(ballot) > 0:
                ballots.append(Ballot(ranking=tuple(ballot), weight=wt_vec[i]))
        return condense_profile(
            PreferenceProfile(
                ballots=tuple(ballots), max_ranking_length=self._ballot_length
            )
        )

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
        self, fpv_vec, wt_vec, w: np.int8, s: int, rng=None
    ) -> np.ndarray:
        """
        Build a list of indices i such that fpv_vec[i] == w;
        each index is repeated round(wt_vec[i]) times, then we sample s entries uniformly.
        This is different from the Cambridge transfer!!!
        Some of the ballots selected for transfer may still be exhausted if they list no further preference.
        """
        if rng is None:
            rng = np.random.default_rng()

        # mask for eligible indices
        mask = fpv_vec == w
        eligible_idx = np.flatnonzero(mask)

        # round weights to nearest int, clip at zero
        weights = np.rint(wt_vec[eligible_idx]).astype(int)
        weights = np.clip(weights, 0, None)

        # build pool of indices
        pool = np.repeat(eligible_idx, weights)

        # sample uniformly from pool
        return rng.choice(pool, size=s, replace=True)

    def __str__(self):
        return self.get_status_df().to_string(index=True, justify="justify")

    __repr__ = __str__


class STV(RankingElection):
    """
    STV elections. All ballots must have no ties.

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        m (int, optional): Number of seats to be elected. Defaults to 1.
        transfer (Callable[[str, float, Union[tuple[Ballot], list[Ballot]], int], tuple[Ballot,...]], optional):
        Transfer method. Defaults to fractional transfer.
            Function signature is elected candidate, their number of first-place votes, the list of
            ballots with them ranked first, and the threshold value. Returns the list of ballots
            after transfer.
        quota (str, optional): Formula to calculate quota. Accepts "droop" or "hare".
            Defaults to "droop".
        simultaneous (bool, optional): True if all candidates who cross threshold in a round are
            elected simultaneously, False if only the candidate with highest first-place votes
            who crosses the threshold is elected in a round. Defaults to True.
        tiebreak (str, optional): Method to be used if a tiebreak is needed. Accepts
            'borda' and 'random'. Defaults to None, in which case a ValueError is raised if
            a tiebreak is needed.

    """  # noqa

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
                    f"Ballot {Ballot(ranking=tuple(row.to_list()), weight = weight_col[idx])} "
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

        elected = []
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
                tiebroken_ranking = tiebreak_set(
                    lowest_fpv_cands, self.get_profile(0), tiebreak="first_place"
                )
                tiebreaks = {lowest_fpv_cands: tiebroken_ranking}

                eliminated_cand = list(tiebroken_ranking[-1])[0]

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
            if self.score_function:
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
        super().__init__(
            profile,
            m=m,
            transfer=(
                lambda winner, fpv, ballots, threshold: tuple(
                    condense_ballot_ranking(remove_cand_from_ballot(winner, b))
                    for b in ballots
                )
            ),
            quota=quota,
            simultaneous=simultaneous,
            tiebreak=tiebreak,
        )
