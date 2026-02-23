import numpy as np
from numpy.typing import NDArray
from votekit.pref_profile import RankProfile
from votekit.elections.election_state import ElectionState
from votekit.utils import tiebreak_set
from typing import Optional, Any
import pandas as pd
from itertools import groupby
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(slots=True)
class NumpyElectionDataTracker:
    """
    Data container for internal use in numpy-based elections.

    Attributes:
        ballot_matrix (NDArray): Matrix of ballot rankings (each row is a unique ballot) with sentinel padding.
        wt_vec (NDArray): Starting weight vector for each ballot row.
        initial_fpv_scores (NDArray): Initial first-preference vote tallies, used for tiebreaking.
        fpv_by_round (list[NDArray]): First-preference tallies by round, used for reporting.
        play_by_play (list[dict[str, Any]]): Per-round action log used for legacy outputs.
        tiebreak_record (list[dict[frozenset[str], tuple[frozenset[str], ...]]]):
            Tiebreak resolutions by round.
        candidate_sets_by_fpv (Optional[list[set[int]]]): Cached FPV clusters for tiebreaking.
        extras (dict[str, Any]): Extension point for child classes to store additional outputs.
    """
    ballot_matrix: NDArray
    wt_vec: NDArray
    initial_fpv_scores: NDArray
    fpv_by_round: list[NDArray] = field(default_factory=list)
    play_by_play: list[dict[str, Any]] = field(default_factory=list)
    tiebreak_record: list[dict[frozenset[str], tuple[frozenset[str], ...]]] = field(
        default_factory=list
    )
    candidate_sets_by_fpv: Optional[list[set[int]]] = None
    extras: dict[str, Any] = field(default_factory=dict)


class NumpySTVBase(ABC):
    candidates: list[str]
    profile: RankProfile
    m: int
    election_states: list[ElectionState]
    threshold: float
    tiebreak: Optional[str]
    _data: NumpyElectionDataTracker
    _winner_tiebreak: Optional[str]
    _loser_tiebreak: Optional[str]

    def __init__(
        self,
        profile: RankProfile,
        m: int = 1,
        tiebreak: Optional[str] = None,
    ):
        self.profile = profile
        self.m = m
        self.candidates = list(profile.candidates)
        self.tiebreak = tiebreak
        self._winner_tiebreak = tiebreak
        self._loser_tiebreak = tiebreak if tiebreak is not None else "first_place"

        ballot_matrix, wt_vec = self._convert_pf_to_numpy_arrays(profile)
        initial_fpv_scores = self._make_initial_fpv(
            np.copy(ballot_matrix[:, 0]), wt_vec
        )
        self._data = NumpyElectionDataTracker(
            ballot_matrix=ballot_matrix,
            wt_vec=wt_vec,
            initial_fpv_scores=initial_fpv_scores,
        )

    @abstractmethod
    def _run_election(
        self, data: NumpyElectionDataTracker
    ) -> tuple[
        list[NDArray],
        list[dict[str, Any]],
        list[dict[frozenset[str], tuple[frozenset[str], ...]]],
    ]:
        """
        Expected outputs: 
        """
        pass

    def _run_and_store(self) -> None:
        """
        Run the election core logic and store results on the data tracker.
        """
        fpv_by_round, play_by_play, tiebreak_record = self._run_election(self._data)
        self._data.fpv_by_round = fpv_by_round
        self._data.play_by_play = play_by_play
        self._data.tiebreak_record = tiebreak_record
        self.election_states = self._make_election_states()

    def get_remaining(self, round_number: int = -1) -> tuple[frozenset, ...]:
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
        tallies = self._data.fpv_by_round[round_number].copy()
        elected_cands_as_list_of_str = [
            c for fset in self.get_elected(round_number) for c in list(fset)
        ]
        elected_cands_numerical = [
            self.candidates.index(c) for c in elected_cands_as_list_of_str
        ]
        tallies[elected_cands_numerical] = 0
        tallies_to_cands = {
            tally: [self.candidates[c] for c, t in enumerate(tallies) if t == tally]
            for tally in tallies
            if tally != 0
        }
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
            round_number < -len(self._data.fpv_by_round)
            or round_number > len(self._data.fpv_by_round) - 1
        ):
            raise IndexError("round_number out of range.")
        round_number = round_number % len(self._data.fpv_by_round)
        list_of_winners = [
            [c]
            for play in self._data.play_by_play[:round_number]
            if play["round_type"] in {"election", "default"}
            for c in play["winners"]
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
            round_number < -len(self._data.fpv_by_round)
            or round_number > len(self._data.fpv_by_round) - 1
        ):
            raise IndexError("round_number out of range.")
        round_number = round_number % len(self._data.fpv_by_round)
        if round_number == 0:
            return tuple()
        list_of_losers = [
            play["loser"]
            for play in self._data.play_by_play[round_number - 1 :: -1]
            if play["round_type"] == "elimination"
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
            round_number < -len(self._data.fpv_by_round)
            or round_number > len(self._data.fpv_by_round) - 1
        ):
            raise IndexError("round_number out of range.")

        round_number = round_number % len(self._data.fpv_by_round)
        new_index = [c for s in self.get_ranking(round_number) for c in s]

        for play in self._data.play_by_play[:round_number]:
            if play["round_type"] == "elimination":
                status_df.at[self.candidates[play["loser"][0]], "Status"] = "Eliminated"
                status_df.at[self.candidates[play["loser"][0]], "Round"] = (
                    play["round_number"] + 1
                )
            elif play["round_type"] in {"election", "default"}:
                for c in play["winners"]:
                    status_df.at[self.candidates[c], "Status"] = "Elected"
                    status_df.at[self.candidates[c], "Round"] = play["round_number"] + 1
        for cand_string in self.candidates:
            if status_df.at[cand_string, "Status"] == "Remaining":
                status_df.at[cand_string, "Round"] = round_number
        status_df = status_df.reindex(new_index)
        return status_df

    def _make_election_states(self):
        """
        Creates the list of election states after the main loop has run.
        Returns:
            list[ElectionState]: List of ElectionState objects representing each round in
                chronological order.
        """
        e_states = [
            ElectionState(
                round_number=0,
                remaining=self.get_remaining(0),
                scores={
                    self.candidates[c]: self._data.fpv_by_round[0][c]
                    for c in self._data.fpv_by_round[0].nonzero()[0]
                },
            )
        ]
        for i, play in enumerate(self._data.play_by_play):
            packaged_tiebreak = self._data.tiebreak_record[i]
            packaged_elected = (
                tuple([frozenset([self.candidates[c]]) for c in play["winners"]])
                if play["round_type"] != "elimination"
                else (frozenset(),)
            )
            packaged_eliminated = (
                (frozenset([self.candidates[c] for c in play["loser"]]),)
                if play["round_type"] == "elimination"
                else (frozenset(),)
            )
            packaged_scores = {
                self.candidates[c]: self._data.fpv_by_round[i + 1][c]
                for c in self._data.fpv_by_round[i + 1].nonzero()[0]
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

    def get_profile(self, round_number: int = -1) -> RankProfile:
        """
        Fetch the RankProfile of the given round number.
        """
        if (
            round_number < -len(self.election_states)
            or round_number > len(self.election_states) - 1
        ):
            raise IndexError("round_number out of range.")

        round_number = round_number % len(self.election_states)

        remaining = self._data.fpv_by_round[round_number].nonzero()[0]
        wt_vec = self._data.wt_vec.copy()
        for i in range(len(self._data.play_by_play[:round_number]) - 1, -1, -1):
            if self._data.play_by_play[i]["round_type"] == "election":
                wt_vec = self._data.play_by_play[i]["wt_vec"]
                break

        idx_to_fset = {c: frozenset([self.candidates[c]]) for c in remaining}

        # --- 1) drop last column by view ---
        A = self._data.ballot_matrix.copy()
        A = A[:, :-1]

        n_rows, n_cols = A.shape

        # --- 2) keep only entries in `remaining` ---
        remaining_arr = np.fromiter((int(x) for x in remaining), dtype=np.int64)
        keep_mask = np.isin(A, remaining_arr)

        # --- 3) stable left-compaction, fill with -127 ---
        out = np.full_like(A, fill_value=-127)  # int8
        pos = keep_mask.cumsum(axis=1) - 1  # target col for each kept entry
        r_idx, c_idx = np.nonzero(keep_mask)
        out[r_idx, pos[r_idx, c_idx]] = A[r_idx, c_idx]

        # --- 3.5) drop rows that are all -127 (empty ballots after filtering) ---
        row_keep_mask = ~(out == -127).all(axis=1)
        out = out[row_keep_mask]
        wt_vec = wt_vec[row_keep_mask]
        n_rows = out.shape[0]

        # --- 3.75) also drop out rows with weight 0 ---
        row_keep_mask = wt_vec != 0
        out = out[row_keep_mask]
        wt_vec = wt_vec[row_keep_mask]
        n_rows = out.shape[0]

        # --- 4) int8 -> frozenset mapping via 256-entry LUT ---
        # default for anything missing (including -127): frozenset("~")
        lut: np.ndarray = np.empty(256, dtype=object)
        lut[:] = frozenset(["~"])
        for k, v in idx_to_fset.items():
            lut[int(np.int16(k)) + 128] = v
        # index into LUT (shift by +128 to map [-128,127] -> [0,255])
        obj = lut[out.astype(np.int16) + 128]  # dtype=object, frozensets

        # --- 5) to DataFrame with Ranking_i columns ---
        data = {f"Ranking_{i+1}": obj[:, i] for i in range(n_cols)}
        df = pd.DataFrame(data)

        # --- 6) Ballot Index column & set as index ---
        df.insert(0, "Ballot Index", np.arange(n_rows, dtype=int))
        df.set_index("Ballot Index", inplace=True)

        # --- 7) Voter Set: empty set per row (distinct objects) ---
        df["Voter Set"] = pd.Series(
            [set() for _ in range(n_rows)], dtype=object, index=df.index
        )

        # --- 8) Weight column ---
        df["Weight"] = wt_vec.astype(np.float64, copy=False)

        return RankProfile(
            max_ranking_length=self.profile.max_ranking_length,
            candidates=tuple([self.candidates[c] for c in remaining]),
            df=df,
        )

    def get_step(self, round_number: int = -1) -> tuple[RankProfile, ElectionState]:
        """
        Fetches the profile and ElectionState of the given round number.

        Args:
            round_number (int): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            tuple[RankProfile, ElectionState]
        """
        return (self.get_profile(round_number), self.election_states[round_number])
    
    def _convert_pf_to_numpy_arrays(self, pf: RankProfile) -> tuple[NDArray, NDArray]:
        """
        This converts the profile into a numpy matrix with some helper arrays for faster iteration.

        Args:
            profile (RankProfile): The preference profile to convert.

        Returns:
            tuple[NDArray, NDArray, NDArray]: The ballot matrix, weights vector, and
                first-preference vector.
        """
        df = pf.df.copy()
        candidate_to_index = {
            frozenset([name]): i for i, name in enumerate(self.candidates)
        }
        candidate_to_index[frozenset(["~"])] = -127

        ranking_columns = [c for c in df.columns if c.startswith("Ranking")]
        num_rows = len(df)
        num_cols = len(ranking_columns)
        if num_cols > len(pf.candidates):
            ranking_columns = ranking_columns[: len(pf.candidates)]
            num_cols = len(ranking_columns)
        cells = df[ranking_columns].to_numpy()

        def map_cell(cell):
            try:
                return candidate_to_index[cell]
            except KeyError:
                raise TypeError(
                    f"Ballots must have rankings, found invalid entry: {cell}"
                )

        mapped = np.frompyfunc(map_cell, 1, 1)(cells).astype(np.int8)

        # Add padding
        ballot_matrix: NDArray = np.full((num_rows, num_cols + 1), -126, dtype=np.int8)
        ballot_matrix[:, :num_cols] = mapped

        wt_vec: NDArray = df["Weight"].astype(np.float64).to_numpy()

        # Reject ballots that have no rankings at all (all -127 or -126)
        empty_rows = np.where(
            np.all((ballot_matrix == -127) | (ballot_matrix == -126), axis=1)
        )[0]
        if empty_rows.size:
            raise TypeError("Ballots must have rankings.")

        return ballot_matrix, wt_vec
    
    def _make_initial_fpv(self, fpv_vec: NDArray, wt_vec: NDArray) -> NDArray:
        """
        Creates the initial first-preference vote (FPV) vector.

        Returns:
            NDArray: The i-th entry is the initial first-preference vote tally for candidate i.
        """
        return np.bincount(
            fpv_vec[fpv_vec != -127],
            weights=wt_vec[fpv_vec != -127],
            minlength=len(self.candidates),
        )

    def _fpv_tiebreak(
        self, tied_cands: list[int], winner_tiebreak_bool: bool
    ) -> tuple[int, tuple[frozenset[str], ...]]:
        """
        Break ties among tied_cands using initial_fpv tallies.

        Args:
            tied_cands (list[int]): List of candidate indices that are tied.
            winner_tiebreak_bool (bool): Whether we are looking for a winner tiebreak (True) or
                loser tiebreak (False).

        Returns:
            tuple: (chosen_candidate_index, packaged_ranking): the candidate index that won the
                winner tiebreak,
                or lost the loser tiebreak, and the packaged tuple of frozensets representing the
                outcome of the tiebreak.
        """

        tied_cands_set = set(tied_cands)
        if self._data.candidate_sets_by_fpv is None:
            scores = np.asarray(self._data.initial_fpv_scores)
            order = np.argsort(scores, kind="mergesort")[::-1]
            pairs = [(float(scores[i]), int(i)) for i in order]
            # Now create a list of sets, not lists
            self._data.candidate_sets_by_fpv = [
                set(idx for _, idx in group)
                for _, group in groupby(pairs, key=lambda x: x[0])
            ]

        clusters_containing_tied_cands: list[set[int]] = [
            cluster & tied_cands_set
            for cluster in self._data.candidate_sets_by_fpv
            if cluster & tied_cands_set
        ]

        packaged_ranking: tuple[frozenset[str], ...] = tuple(
            frozenset(self.candidates[i] for i in cluster)
            for cluster in clusters_containing_tied_cands
        )

        relevant = 0 if winner_tiebreak_bool else -1
        target_cluster = list(clusters_containing_tied_cands[relevant])

        if len(target_cluster) == 1:
            return target_cluster[0], packaged_ranking

        tiebroken_candidate = int(np.random.choice(target_cluster))
        return tiebroken_candidate, packaged_ranking

    def _get_threshold(
        self,
        quota_type,
        total_ballot_wt: float,
        floor: bool = True,
        epsilon: float = 1.0,
    ) -> float:
        """
        Calculates threshold required for election.

        Args:
            total_ballot_wt (float): Total weight of ballots to compute threshold.
        Returns:
            float: Value of the threshold.
        """
        if quota_type == "droop":
            fractional_quota = total_ballot_wt / (self.m + 1)
            if floor:
                return int(fractional_quota) + epsilon
            else:
                return fractional_quota + epsilon
        elif quota_type == "hare":
            fractional_quota = total_ballot_wt / self.m
            if floor:
                return int(fractional_quota)
            else:
                return fractional_quota
        else:
            raise ValueError("Misspelled or unknown quota type.")

    def _run_winner_tiebreak(
        self,
        tied_winners: list[int],
        turn: int,
        mutant_tiebreak_record: list[dict[frozenset[str], tuple[frozenset[str], ...]]],
    ) -> tuple[int, list[dict[frozenset[str], tuple[frozenset[str], ...]]]]:
        """
        Handle winner tiebreaking logic.

        Args:
            tied_winners (list[int]): List of candidate indices that are tied.
            turn (int): The current round number.
            mutant_tiebreak_record (list[dict[frozenset[str], tuple[frozenset[str], ...]]]):
                Tiebreak record for each round.

        Returns:
            tuple: (index of new winner, updated tiebreak record)
        """
        packaged_tie = frozenset(
            [self.candidates[winner_idx] for winner_idx in tied_winners]
        )
        if self._winner_tiebreak == "first_place":
            winner_idx, packaged_ranking = self._fpv_tiebreak(
                tied_winners, winner_tiebreak_bool=True
            )
        elif self._winner_tiebreak is not None:
            packaged_ranking = tiebreak_set(
                r_set=packaged_tie,
                profile=self.profile,
                tiebreak=self._winner_tiebreak,
            )
            winner_idx = self.candidates.index(list(packaged_ranking[0])[0])
        else:
            raise ValueError(
                "Cannot elect correct number of candidates without breaking ties."
            )
        mutant_tiebreak_record.append({packaged_tie: packaged_ranking})
        return winner_idx, mutant_tiebreak_record

    def _run_loser_tiebreak(
        self,
        tied_losers: list[int],
        turn: int,
        mutant_tiebreak_record: list[dict[frozenset[str], tuple[frozenset[str], ...]]],
    ) -> tuple[int, list[dict[frozenset[str], tuple[frozenset[str], ...]]]]:
        """
        Handle loser tiebreaking logic.

        Args:
            tied_losers (list[int]): List of candidate indices that are tied.
            turn (int): The current round number.
            mutant_tiebreak_record (list[dict[frozenset[str], tuple[frozenset[str], ...]]]):
                Tiebreak record for each round.

        Returns:
            tuple: (index of new loser, updated tiebreak record)
        """
        packaged_tie = frozenset(
            [self.candidates[winner_idx] for winner_idx in tied_losers]
        )
        if self._loser_tiebreak == "first_place":
            loser_idx, packaged_ranking = self._fpv_tiebreak(
                tied_losers, winner_tiebreak_bool=False
            )
        else:
            packaged_ranking = tiebreak_set(
                r_set=packaged_tie,
                profile=self.profile,
                tiebreak=self._loser_tiebreak,
            )
            loser_idx = self.candidates.index(list(packaged_ranking[-1])[0])
        mutant_tiebreak_record.append({packaged_tie: packaged_ranking})
        return loser_idx, mutant_tiebreak_record

    def __str__(self):
        return self.get_status_df().to_string(index=True, justify="justify")
    
    __repr__ = __str__


class NumpyElection(NumpySTVBase):
    """
    Legacy shim to keep FastSTV compatible with the previous NumpyElection interface.
    """

    def _run_election(
        self, data: NumpyElectionDataTracker
    ) -> tuple[
        list[NDArray],
        list[dict[str, Any]],
        list[dict[frozenset[str], tuple[frozenset[str], ...]]],
    ]:
        return self._core.run()


class ElectionCore:
    def __init__(
        self,
        profile: RankProfile,
        m: int = 1,
        candidates: list[str] = [],
        tiebreak: Optional[str] = None,
    ):
        self.candidates = (
            candidates if len(candidates) > 0 else list(profile.candidates)
        )
        self._ballot_matrix, self._wt_vec = self._convert_pf_to_numpy_arrays(profile)
        self.profile = profile
        self.m = m
        self._winner_tiebreak = tiebreak
        self._loser_tiebreak = tiebreak if tiebreak is not None else "first_place"
        self._initial_fpv_scores = self._make_initial_fpv(
            np.copy(self._ballot_matrix[:, 0])
        )

    def _convert_pf_to_numpy_arrays(self, pf: RankProfile) -> tuple[NDArray, NDArray]: #✔
        """
        This converts the profile into a numpy matrix with some helper arrays for faster iteration.

        Args:
            profile (RankProfile): The preference profile to convert.

        Returns:
            tuple[NDArray, NDArray, NDArray]: The ballot matrix, weights vector, and
                first-preference vector.
        """
        df = pf.df.copy()
        candidate_to_index = {
            frozenset([name]): i for i, name in enumerate(self.candidates)
        }
        candidate_to_index[frozenset(["~"])] = -127

        ranking_columns = [c for c in df.columns if c.startswith("Ranking")]
        num_rows = len(df)
        num_cols = len(ranking_columns)
        if num_cols > len(pf.candidates):
            ranking_columns = ranking_columns[: len(pf.candidates)]
            num_cols = len(ranking_columns)
        cells = df[ranking_columns].to_numpy()

        def map_cell(cell):
            try:
                return candidate_to_index[cell]
            except KeyError:
                raise TypeError(
                    f"Ballots must have rankings, found invalid entry: {cell}"
                )

        mapped = np.frompyfunc(map_cell, 1, 1)(cells).astype(np.int8)

        # Add padding
        ballot_matrix: NDArray = np.full((num_rows, num_cols + 1), -126, dtype=np.int8)
        ballot_matrix[:, :num_cols] = mapped

        wt_vec: NDArray = df["Weight"].astype(np.float64).to_numpy()

        # Reject ballots that have no rankings at all (all -127 or -126)
        empty_rows = np.where(
            np.all((ballot_matrix == -127) | (ballot_matrix == -126), axis=1)
        )[0]
        if empty_rows.size:
            raise TypeError("Ballots must have rankings.")

        return ballot_matrix, wt_vec

    def _make_initial_fpv(self, fpv_vec) -> NDArray:
        """
        Creates the initial first-preference vote (FPV) vector.

        Returns:
            NDArray: The i-th entry is the initial first-preference vote tally for candidate i.
        """
        return np.bincount(
            fpv_vec[fpv_vec != -127],
            weights=self._wt_vec[fpv_vec != -127],
            minlength=len(self.candidates),
        )

    def _fpv_tiebreak(
        self, tied_cands: list[int], winner_tiebreak_bool: bool
    ) -> tuple[int, tuple[frozenset[str], ...]]:
        """
        Break ties among tied_cands using initial_fpv tallies.

        Args:
            tied_cands (list[int]): List of candidate indices that are tied.
            winner_tiebreak_bool (bool): Whether we are looking for a winner tiebreak (True) or
                loser tiebreak (False).

        Returns:
            tuple: (chosen_candidate_index, packaged_ranking): the candidate index that won the
                winner tiebreak,
                or lost the loser tiebreak, and the packaged tuple of frozensets representing the
                outcome of the tiebreak.
        """

        tied_cands_set = set(tied_cands)
        if not hasattr(self, "__candidate_sets_by_fpv"):
            scores = np.asarray(self._initial_fpv_scores)
            order = np.argsort(scores, kind="mergesort")[::-1]
            pairs = [(float(scores[i]), int(i)) for i in order]
            # Now create a list of sets, not lists
            self.__candidate_sets_by_fpv: list[set[int]] = [
                set(idx for _, idx in group)
                for _, group in groupby(pairs, key=lambda x: x[0])
            ]

        clusters_containing_tied_cands: list[set[int]] = [
            cluster & tied_cands_set
            for cluster in self.__candidate_sets_by_fpv
            if cluster & tied_cands_set
        ]

        packaged_ranking: tuple[frozenset[str], ...] = tuple(
            frozenset(self.candidates[i] for i in cluster)
            for cluster in clusters_containing_tied_cands
        )

        relevant = 0 if winner_tiebreak_bool else -1
        target_cluster = list(clusters_containing_tied_cands[relevant])

        if len(target_cluster) == 1:
            return target_cluster[0], packaged_ranking

        tiebroken_candidate = int(np.random.choice(target_cluster))
        return tiebroken_candidate, packaged_ranking

    def _get_threshold(
        self,
        quota_type,
        total_ballot_wt: float,
        floor: bool = True,
        epsilon: float = 1.0,
    ) -> float:
        """
        Calculates threshold required for election.

        Args:
            total_ballot_wt (float): Total weight of ballots to compute threshold.
        Returns:
            float: Value of the threshold.
        """
        if quota_type == "droop":
            fractional_quota = total_ballot_wt / (self.m + 1)
            if floor:
                return int(fractional_quota) + epsilon
            else:
                return fractional_quota + epsilon
        elif quota_type == "hare":
            fractional_quota = total_ballot_wt / self.m
            if floor:
                return int(fractional_quota)
            else:
                return fractional_quota
        else:
            raise ValueError("Misspelled or unknown quota type.")

    def _run_winner_tiebreak(
        self,
        tied_winners: list[int],
        turn: int,
        mutant_tiebreak_record: list[dict[frozenset[str], tuple[frozenset[str], ...]]],
    ) -> tuple[int, list[dict[frozenset[str], tuple[frozenset[str], ...]]]]:
        """
        Handle winner tiebreaking logic.

        Args:
            tied_winners (list[int]): List of candidate indices that are tied.
            turn (int): The current round number.
            mutant_tiebreak_record (list[dict[frozenset[str], tuple[frozenset[str], ...]]]):
                Tiebreak record for each round.

        Returns:
            tuple: (index of new winner, updated tiebreak record)
        """
        packaged_tie = frozenset(
            [self.candidates[winner_idx] for winner_idx in tied_winners]
        )
        if self._winner_tiebreak == "first_place":
            winner_idx, packaged_ranking = self._fpv_tiebreak(
                tied_winners, winner_tiebreak_bool=True
            )
        elif self._winner_tiebreak is not None:
            packaged_ranking = tiebreak_set(
                r_set=packaged_tie, profile=self.profile, tiebreak=self._winner_tiebreak
            )
            winner_idx = self.candidates.index(list(packaged_ranking[0])[0])
        else:
            raise ValueError(
                "Cannot elect correct number of candidates without breaking ties."
            )
        mutant_tiebreak_record.append({packaged_tie: packaged_ranking})
        return winner_idx, mutant_tiebreak_record

    def _run_loser_tiebreak(
        self,
        tied_losers: list[int],
        turn: int,
        mutant_tiebreak_record: list[dict[frozenset[str], tuple[frozenset[str], ...]]],
    ) -> tuple[int, list[dict[frozenset[str], tuple[frozenset[str], ...]]]]:
        """
        Handle loser tiebreaking logic.

        Args:
            tied_losers (list[int]): List of candidate indices that are tied.
            turn (int): The current round number.
            mutant_tiebreak_record (list[dict[frozenset[str], tuple[frozenset[str], ...]]]):
                Tiebreak record for each round.

        Returns:
            tuple: (index of new loser, updated tiebreak record)
        """
        packaged_tie = frozenset(
            [self.candidates[winner_idx] for winner_idx in tied_losers]
        )
        if self._loser_tiebreak == "first_place":
            loser_idx, packaged_ranking = self._fpv_tiebreak(
                tied_losers, winner_tiebreak_bool=False
            )
        else:
            packaged_ranking = tiebreak_set(
                r_set=packaged_tie, profile=self.profile, tiebreak=self._loser_tiebreak
            )
            loser_idx = self.candidates.index(list(packaged_ranking[-1])[0])
        mutant_tiebreak_record.append({packaged_tie: packaged_ranking})
        return loser_idx, mutant_tiebreak_record
