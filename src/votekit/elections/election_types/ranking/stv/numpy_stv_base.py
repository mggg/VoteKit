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
from enum import Enum


class NumpySTVSentinel(Enum):
    """
    Enum for sentinel values in the ballot matrix.

    Attributes:
        BLANK_RANKING (int): Sentinel value for blank/empty rankings after padding.
            This value is used at the end of the ballot matrix to indicate the end of each ballot,
            and candidates are replaced with this value in the ballot matrix
            when they are elected or eliminated.
    """

    BLANK_RANKING = np.int8(-127)


@dataclass(slots=True)
class NumpyElectionDataTracker:
    """
    Data container for internal use in numpy-based elections.

    Attributes:
        ballot_matrix (NDArray): Matrix of ballots (each row is a unique ballot) with sentinel padding.
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
    """
    Abstract base class for numpy-based STV-style elections.

    Attributes:
        candidates (list[str]): List of candidate names, indexed
            to correspond to ballot matrix entries.
        profile (RankProfile): The original RankProfile for reference.
        m (int): Number of seats to be elected.
        election_states (list[ElectionState]): List of ElectionState objects representing each round in
            chronological order.
        tiebreak (Optional[str]): User-specified method to be used if a tiebreak is needed.
            Defaults to None.
        _data (NumpyElectionDataTracker): Internal data tracker.
        _winner_tiebreak (Optional[str]): Tiebreak method for winners, set to `None` by default.
        _loser_tiebreak (str): Tiebreak method for losers, set to "first_place" by default.
    """

    candidates: list[str]
    profile: RankProfile
    m: int
    election_states: list[ElectionState]
    threshold: float
    tiebreak: Optional[str]
    _data: NumpyElectionDataTracker
    _winner_tiebreak: Optional[str]
    _loser_tiebreak: str

    def __init__(
        self,
        profile: RankProfile,
        m: int = 1,
        tiebreak: Optional[str] = None,
    ):
        """
        Args:
            profile (RankProfile): RankProfile to run election on.
            m (int): Number of seats to be elected. Defaults to 1.
            tiebreak (Optional[str]): Method to be used if a tiebreak is needed. Defaults to None.

        Returns:
            NumpySTVBase: Initialized base instance.
        """
        self.profile = profile
        self.m = m
        self.candidates = list(profile.candidates)
        self.tiebreak = tiebreak
        self._winner_tiebreak = tiebreak
        self._loser_tiebreak = tiebreak if tiebreak is not None else "first_place"

        ballot_matrix, wt_vec = self._convert_profile_to_numpy_arrays(profile)
        initial_fpv_scores = self._make_initial_fpv(
            np.copy(ballot_matrix[:, 0]), wt_vec
        )
        self._data = NumpyElectionDataTracker(
            ballot_matrix=ballot_matrix,
            wt_vec=wt_vec,
            initial_fpv_scores=initial_fpv_scores,
        )

    # ==================
    # == Init Methods ==
    # ==================

    def _convert_profile_to_numpy_arrays(
        self, pf: RankProfile
    ) -> tuple[NDArray, NDArray]:
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
        candidate_to_index[frozenset(["~"])] = int(
            NumpySTVSentinel.BLANK_RANKING.value
        )  # sentinel for blank/empty rankings after padding

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
                raise TypeError(f"Found invalid entry: {cell}")

        mapped = np.frompyfunc(map_cell, 1, 1)(cells).astype(np.int8)

        # Add padding -- a lot of the election logic needs at least one entry of each row of the ballot matrix to be negative.
        # Specifically, the bool_ballot_matrix is initialized as all 1s, and its entries are set to 0 only when candidates are eliminated/elected.
        # We use an argmax on the bool_ballot_matrix to find the next preference for each ballot, which relies on having at least one entry in each row be 0.
        ballot_matrix: NDArray = np.full(
            (num_rows, num_cols + 1),
            NumpySTVSentinel.BLANK_RANKING.value,
            dtype=np.int8,
        )
        ballot_matrix[:, :num_cols] = mapped

        wt_vec: NDArray = df["Weight"].astype(np.float64).to_numpy()

        return ballot_matrix, wt_vec

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

    def _make_initial_fpv(self, fpv_vec: NDArray, wt_vec: NDArray) -> NDArray:
        """
        Creates the initial first-preference vote (FPV) vector.

        Returns:
            NDArray: The i-th entry is the initial first-preference vote tally for candidate i.
        """
        return np.bincount(
            fpv_vec[fpv_vec != NumpySTVSentinel.BLANK_RANKING.value],
            weights=wt_vec[fpv_vec != NumpySTVSentinel.BLANK_RANKING.value],
            minlength=len(self.candidates),
        )

    @abstractmethod
    def _run_election(self, data: NumpyElectionDataTracker) -> tuple[
        list[NDArray],
        list[dict[str, Any]],
        list[dict[frozenset[str], tuple[frozenset[str], ...]]],
    ]:
        """
        Core election logic to be implemented by child classes.

        This should run the election and produce the outputs needed to populate the
        NumpyElectionDataTracker. Note that the child class can store additional outputs
        in the `extras` field of the data tracker if needed.

        Args:
            data (NumpyElectionDataTracker): The initialized data tracker with the profile converted to
                numpy arrays.

        Returns:
            fpv_by_round (list[NDArray]): List of first-preference vote tallies by round.
            play_by_play (list[dict[str, Any]]): List of dictionaries representing the actions taken in each
                round.
            tiebreak_record (list[dict[frozenset[str], tuple[frozenset[str], ...]]]): List of dictionaries
                representing tiebreak resolutions for each round.
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

    # ==================
    # == User Methods ==
    # ==================

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
            [c for c in play["winners"]]
            for play in self._data.play_by_play[:round_number]
            if play["round_type"] in {"election", "default"}
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
        Returns a dataframe reporting the status of the candidates in the given round.

        DataFrame is sorted by current ranking.

        Args:
            round_number (int): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            pd.DataFrame: Dataframe displaying candidate, status (elected, eliminated, remaining), and the
                round their status updated.
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

    def get_profile(self, round_number: int = -1) -> RankProfile:
        """
        Returns the RankProfile of the given round number.
        """
        if (
            round_number < -len(self.election_states)
            or round_number > len(self.election_states) - 1
        ):
            raise IndexError("round_number out of range.")

        round_number = round_number % len(self.election_states)

        remaining = self._data.fpv_by_round[round_number].nonzero()[0]
        # start from stored weights; may be overwritten by play-by-play for this round
        wt_vec = self._data.wt_vec
        for i in range(len(self._data.play_by_play[:round_number]) - 1, -1, -1):
            if self._data.play_by_play[i]["round_type"] == "election":
                wt_vec = self._data.play_by_play[i]["wt_vec"]
                break

        idx_to_fset = {c: frozenset([self.candidates[c]]) for c in remaining}

        # --- 1) drop last column by view (sentinel column) ---
        A = self._data.ballot_matrix[:, :-1]

        n_rows, n_cols = A.shape

        # --- 2) keep only entries in `remaining` ---
        remaining_arr = np.fromiter((int(x) for x in remaining), dtype=np.int64)
        keep_mask = np.isin(A, remaining_arr)

        # --- 3) stable left-compaction, fill with -127 ---
        out = np.full_like(A, fill_value=NumpySTVSentinel.BLANK_RANKING.value)  # int8
        pos = keep_mask.cumsum(axis=1) - 1  # target col for each kept entry
        r_idx, c_idx = np.nonzero(keep_mask)
        out[r_idx, pos[r_idx, c_idx]] = A[r_idx, c_idx]

        # --- 3.5) drop rows that are empty after filtering AND rows with weight 0 ---
        # keep rows that have at least one remaining candidate and nonzero weight
        row_keep_mask = ~(out == NumpySTVSentinel.BLANK_RANKING.value).all(axis=1) & (
            wt_vec != 0
        )
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
        obj = lut[out.astype(np.int16, copy=False) + 128]  # dtype=object, frozensets

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
        Returns the profile and ElectionState of the given round number.

        Args:
            round_number (int): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            tuple[RankProfile, ElectionState]
        """
        return (self.get_profile(round_number), self.election_states[round_number])

    # ======================
    # == Internal Methods ==
    # ======================

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
        round_number: int,
        mutant_tiebreak_record: list[dict[frozenset[str], tuple[frozenset[str], ...]]],
    ) -> tuple[int, list[dict[frozenset[str], tuple[frozenset[str], ...]]]]:
        """
        Handle winner tiebreaking logic.

        Args:
            tied_winners (list[int]): List of candidate indices that are tied.
            round_number (int): The current round number.
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
        round_number: int,
        mutant_tiebreak_record: list[dict[frozenset[str], tuple[frozenset[str], ...]]],
    ) -> tuple[int, list[dict[frozenset[str], tuple[frozenset[str], ...]]]]:
        """
        Handle loser tiebreaking logic.

        Args:
            tied_losers (list[int]): List of candidate indices that are tied.
            round_number (int): The current round number.
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
