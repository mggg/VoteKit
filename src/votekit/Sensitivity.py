"""
Sensitivity analysis for social choice (voting) data using ballot perturbation.
Includes dropout and jitter simulations for various voting types.

Supported voting_method options:
    - 'pickone':     Ballot is a single choice per voter (plurality).
    - 'pickn':       Ballot allows picking :math:`n` candidates.
    - 'rate':        Ballot contains numerical ratings/scores for candidates.
    - 'rank':        Ballot is an ordered preference ranking.
    - 'accept_approve': Accept/approve-like or anti-plurality method, where voters assign first-choice (1), acceptable (2), or reject (other) to candidates.
    - 'approval':    Ballot is approval-style (approve/disapprove candidates).

Classes:
    - SensitivityTestBaseClass: Base for all sensitivity tests.
    - DropoutTest: Ballot dropout simulation for robustness.
    - JitterTest: Jitter/noise simulation for ballot entries.
 """

import numpy as np
import pandas as pd
from votekit.elections import IRV
from votekit.pref_profile import PreferenceProfile
from votekit.ballot import Ballot
from abc import abstractmethod
from votekit.sensitivity_utils import array_to_ballots, ballots_to_array, generate_irv_ballots

ballots_to_array = np.vectorize(ballots_to_array, cache=True)
array_to_ballots = np.vectorize(array_to_ballots, cache=True, signature="(n)->()")

class SensitivityTestBaseClass:
    """
    Base class for sensitivity analysis on voting data.

    Args:
        df (pd.DataFrame | np.ndarray | list[list]): Voting data (ballots or ratings).
        n_iters (int): Number of random simulation runs per test, :math:`n_{\\text{iters}}`.
        volume (float): Fraction of data (ballots, entries, or votes) to affect each iteration, in :math:`[0, 1]`.
        n_ranking (int, optional): Number of top :math:`n` places/candidates to record. Defaults to number of candidates.
        labels (list[str] | np.ndarray, optional): Candidate names or identifiers. Required for ndarray/list input.
        voting_method (str): One of ['pickone', 'pickn', 'rate', 'rank', 'accept_approve', 'approval'].
            Determines ballot and perturbation type.
        max_rank (int, optional): Maximum allowed ranking position/places to sample, :math:`\\text{max_rank}`.
    Attributes:
        df (np.ndarray): Numeric data copy for perturbation.
        labels (np.ndarray): Candidate names or identifiers.
        n_ranking (int): Number of top places to track, :math:`n_{\\text{ranking}}`.
        volume (float): Dropout/jitter intensity, :math:`v \\in [0, 1]`.
        total_volume (int): Total possible sample units, calculated based on voting_method (e.g., ballots × candidates for 'pickone', or sum of maximum scores for 'rate').
        sample_n (int): Number of units to perturb, :math:`\\text{sample_n} = \\text{total_volume} \\times v`.
    Raises:
        ValueError: If voting_method is not valid.
        TypeError: If ndarray/list input is missing labels.
    """
    def __init__(
        self,
        df: pd.DataFrame | np.ndarray | list[list] = None,
        n_iters: int = 10_000,
        volume: float = 0.95,
        n_ranking: int = None,
        labels: list[str] | np.ndarray = None,
        voting_method: str = "pickone",
        max_rank: int = 5,
    ) -> None:
        self.df = df
        if isinstance(df, list):
            self.df = np.array(self.df)
        if isinstance(df, pd.DataFrame):
            self.labels = np.array(df.columns)
            self.df = self.df.to_numpy()
        elif isinstance(df, (np.ndarray, list)):
            if labels is None:
                raise TypeError(
                    "if df is numpy ndarray or list, labels should not be None"
                )
            else:
                self.labels = np.array(labels)
            self.df = self.df.astype(np.float64)
        self.max_rank = max_rank
        self.n_iters = n_iters
        self.volume = volume
        self.n_ranking = n_ranking if n_ranking is not None else self.df.shape[1]
        self.ranking_iter_list = []
        self.voting_method = voting_method

        valid_voting_methods = ["pickone", "pickn", "rate", "rank", "accept_approve", "approval"]
        if self.voting_method not in valid_voting_methods:
            raise ValueError(
                f"invalid voting method, use one of {valid_voting_methods}"
            )

        # Volume calculations by voting method:
        # - 'pickone': Total ballots times candidates.
        # - 'rate': Total ballots times candidates times maximum score.
        # - 'accept_approve': Total ballots times candidates times 3 (for first-choice, acceptable, reject).
        # - 'approval': Total ballots times candidates times 2 (for approve/disapprove).
        # - 'rank': Sum of ranks from 1 to min(max_rank, candidates).
        if self.voting_method == "pickone":
            self.total_volume = self.df.shape[0] * self.df.shape[1]
        elif self.voting_method == "rate":
            self.total_volume = (
                self.df.shape[0]
                * self.df.shape[1]
                * int(np.max(self.df[~np.isnan(self.df)]))
            )
        elif self.voting_method == "accept_approve":
            self.total_volume = self.df.shape[0] * self.df.shape[1] * 3
        elif self.voting_method == "approval":
            self.total_volume = self.df.shape[0] * self.df.shape[1] * 2
        elif self.voting_method == "rank":
            self.total_volume = self.df.shape[0] * np.sum(
                np.arange(
                    self.df.shape[1] - np.min([self.max_rank, self.df.shape[1]]) + 1,
                    self.df.shape[1] + 1,
                )
            )
        self.sample_n = int(self.total_volume * self.volume)

    @abstractmethod
    def run(self):
        """
        Abstract method for executing the sensitivity simulation.

        Raises:
            NotImplementedError: Always, as a stub.
        """
        raise NotImplementedError

    def add_to_result(self, curiter_df: np.ndarray) -> None:
        """
        Compute ranking for the current perturbed matrix and append to results list.

        Args:
            curiter_df (np.ndarray): The perturbed data for this iteration.
        """
        rank_order = np.nan_to_num(curiter_df).sum(axis=0).argsort()[::-1]
        self.ranking_iter_list.append(self.labels[rank_order][: self.n_ranking])

    def add_to_result_accept_approve(self, curiter_df: np.ndarray) -> None:
        """
        For accept/approve voting, select first-choice winner and order the rest.

        Args:
            curiter_df (np.ndarray): The perturbed data for this iteration.
        """
        fc_idx = np.argmax((curiter_df == 1.0).sum(axis=0))
        fc = self.labels[fc_idx]
        rest_idx = (
            ((curiter_df == 2.0) | (curiter_df == 1.0)).sum(axis=0).argsort()[::-1]
        )
        self.ranking_iter_list.append(
            [fc] + self.labels[rest_idx[rest_idx != fc_idx]].tolist()
        )

    def add_to_result_irv(self, ballots: list[Ballot]) -> None:
        """
        For ranked voting, run IRV on the current ballot set and record rankings.

        Args:
            ballots (list[votekit.ballot.Ballot]): List of Ballot objects for this iteration.
        """
        irv_profile = PreferenceProfile(ballots=ballots)
        irv_election = IRV(profile=irv_profile)
        irv_rankings = irv_election.run_election().rankings()
        irv_places = []
        for ranking in irv_rankings:
            irv_places += list(ranking)
        self.ranking_iter_list.append(irv_places)

    def _result(self) -> pd.DataFrame:
        """
        Tabulate and summarize result rankings observed over all iterations.

        Returns:
            pd.DataFrame: Table of distinct observed rankings and their proportion frequency.
        """
        result_df = (
            pd.DataFrame(self.ranking_iter_list)
            .value_counts(normalize=True)
            .reset_index(allow_duplicates=True)
        )
        result_df.columns = result_df.columns[:-1].to_list() + ["proportion"]
        result_df.columns = [
            f"Place {int(col)+1}" for col in result_df.columns[: self.n_ranking]
        ] + result_df.columns[self.n_ranking :].to_list()
        result_df.loc[:, "iters"] = self.n_iters
        result_df.loc[:, "volume"] = self.volume
        return result_df

class DropoutTest(SensitivityTestBaseClass):
    """
    Sensitivity test for simulating dropout or missing votes/ballots.

    Operates on various ballot types ('pickone', 'rank', 'rate', 'approval', 'accept_approve').
    Each iteration simulates removal of a subset of voters, entries, or ballot entries.

    Args:
        df (pd.DataFrame | np.ndarray | list[list]): Ballot data matrix.
        n_iters (int): Number of dropout iterations, :math:`n_{\\text{iters}}`.
        volume (float): Proportion of total sample units dropped, in :math:`[0, 1]`.
        n_ranking (int, optional): Number of top :math:`n` positions to record. Defaults to number of candidates.
        labels (list[str] | np.ndarray, optional): Candidate names or identifiers.
        voting_method (str): Type of votes, one of ['pickone', 'pickn', 'rate', 'rank', 'accept_approve', 'approval'].
    Returns:
        pd.DataFrame: Tabulated frequency of each observed ranking across runs.
    """
    def __init__(
        self,
        df: pd.DataFrame | np.ndarray | list[list] = None,
        n_iters: int = 10000,
        volume: float = 0.05,
        n_ranking: int = None,
        labels: list[str] | np.ndarray = None,
        voting_method: str = "pickone",
    ) -> None:
        super().__init__(df, n_iters, volume, n_ranking, labels, voting_method)

    def run(self, volume: float = None) -> pd.DataFrame:
        """
        Execute the dropout sensitivity simulation.

        For each iteration, randomly drops ballots or ballot entries according to voting_method,
        recalculates outcomes, and tabulates results.

        Args:
            volume (float, optional): Override intensity of dropout for this run, in :math:`[0, 1]`.

        Returns:
            pd.DataFrame: Summary of observed rankings and their proportions.
        """
        if volume is not None:
            self.volume = volume
            self.sample_n = int(self.total_volume * self.volume)

        self.ranking_iter_list = []
        rng = np.random.default_rng()

        if self.voting_method == "rank":
            self.ballots = np.array(
                generate_irv_ballots(
                    pd.DataFrame(data=self.df, columns=self.labels), self.labels
                )
            )
            for _ in range(self.n_iters):
                sample_indices = rng.choice(
                    len(self.ballots),
                    size=int(len(self.ballots) * (1 - self.volume)),
                    replace=False,
                )
                cur_ballots = self.ballots[sample_indices].tolist()
                self.add_to_result_irv(cur_ballots)

        elif self.voting_method == "accept_approve":
            for _ in range(self.n_iters):
                curiter_df = np.copy(self.df)
                rows = rng.choice(self.df.shape[0], size=int(self.sample_n / 3))
                cols = rng.choice(self.df.shape[1], size=int(self.sample_n / 3))
                curiter_df[rows, cols] = np.nan
                self.add_to_result_accept_approve(curiter_df)

        elif self.voting_method == "pickone":
            for _ in range(self.n_iters):
                curiter_df = np.copy(self.df)
                rows = rng.choice(
                    self.df.shape[0],
                    size=self.df.shape[0] - int(self.sample_n / self.df.shape[1]),
                    replace=False
                )
                if len(rows) == 0:
                    self.add_to_result(self.df)
                else:
                    curiter_df = curiter_df[rows]
                    self.add_to_result(curiter_df)

        elif self.voting_method == "rate" or self.voting_method == "approval":
            for _ in range(self.n_iters):
                curiter_df = np.copy(self.df)
                rows = rng.choice(self.df.shape[0], size=self.sample_n)
                cols = rng.choice(self.df.shape[1], size=self.sample_n)
                curiter_df[rows, cols] = np.nan
                self.add_to_result(curiter_df)

        return self._result()

class JitterTest(SensitivityTestBaseClass):
    """
    Sensitivity test for simulating random 'jitter' (noise) in voter preferences.

    For each voting method, jitter is defined as:
        - 'pickone' / 'pickn': Shuffle a random subset of places on ballots.
        - 'rate': Randomly increase/decrease ballot entries by rate_move, respecting bounds.
        - 'rank': Swap ballot positions for random candidates.
        - 'accept_approve': Perturb categorical first-choice/acceptable/reject choices.
        - 'approval': Flip approval status of select entries.

    Args:
        df (pd.DataFrame | np.ndarray | list[list]): Ballot data or vote matrix.
        n_iters (int): Number of jitter iterations, :math:`n_{\\text{iters}}`.
        volume (float): Proportion of sample entries jittered in each iteration, in :math:`[0, 1]`.
        n_ranking (int, optional): Number of top :math:`n` places to record. Defaults to number of candidates.
        labels (list[str] | np.ndarray, optional): Candidate names or identifiers.
        voting_method (str): Ballot type, one of ['pickone', 'pickn', 'rate', 'rank', 'accept_approve', 'approval'].
        n_to_change (int, optional): Entries per ballot to jitter. Defaults to number of candidates.
        rate_max (float, optional): Maximum score for 'rate' ballots, :math:`r_{\\text{max}}`.
        rate_move (float, optional): Step size for 'rate' ballot changes, :math:`\\Delta r`.

    Returns:
        pd.DataFrame: Table of observed rankings/winners and their proportions.
    """
    def __init__(
        self,
        df: pd.DataFrame | np.ndarray | list[list] = None,
        n_iters: int = 1000,
        volume: float = 0.05,
        n_ranking: int = None,
        labels: list[str] | np.ndarray = None,
        voting_method: str = "pickone",
        n_to_change: int = None,
        rate_max: float = 3.0,
        rate_move: float = 1.0,
    ) -> None:
        super().__init__(df, n_iters, volume, n_ranking, labels, voting_method)
        if n_to_change is not None:
            self.n_to_change = n_to_change
        else:
            self.n_to_change = self.df.shape[1]
        self.rate_max = rate_max
        self.rate_move = rate_move

    def run(self, volume: float = None) -> pd.DataFrame:
        """
        Executes the jitter sensitivity simulation.

        For each iteration, randomly perturbs a subset of ballot entries according
        to voting_method, recalculates rankings or winners, and collects results.

        Args:
            volume (float, optional): Override jitter volume for this call, in :math:`[0, 1]`.

        Returns:
            pd.DataFrame: Summary of observed rankings over all iterations.
        """
        if volume is not None:
            self.volume = volume
            self.sample_n = int(self.total_volume * self.volume)
        self.ranking_iter_list = []

        if self.voting_method == "pickone":
            self._run_pickone()
        elif self.voting_method == "pickn":
            self._run_pickone()
        elif self.voting_method == "rate":
            self._run_rate()
        elif self.voting_method == "rank":
            self._run_rank()
        elif self.voting_method == "accept_approve":
            self._run_aa()
        elif self.voting_method == "approval":
            self._run_approval()
        return self._result()

    def _run_pickone(self) -> None:
        """
        Jitter for 'pickone' and 'pickn': for random subset of impor, permute candidate assignment.
        """
        rng = np.random.default_rng()
        for _ in range(self.n_iters):
            curiter_df = np.copy(self.df)
            sample_indices = rng.choice(
                len(self.df), size=int(self.sample_n / self.df.shape[1]), replace=True
            )
            to_jitter = curiter_df[sample_indices]
            jittered = rng.permuted(to_jitter, axis=1)
            curiter_df[sample_indices] = jittered
            self.add_to_result(curiter_df)

    def _run_rate(self) -> None:
        """
        Jitter for 'rate' voting: random +/- rate_move perturbation of scores, with bounds.
        """
        rng = np.random.default_rng()
        for _ in range(self.n_iters):
            curiter_df = np.copy(self.df)
            rows = rng.choice(self.df.shape[0], size=self.sample_n)
            cols = rng.choice(self.df.shape[1], size=self.sample_n)
            rc_df = (
                pd.DataFrame([rows, cols]).T.value_counts().reset_index(name="count")
            )
            while rc_df.shape[0] > 0:
                rows = rc_df.iloc[:, 0].to_numpy()
                cols = rc_df.iloc[:, 1].to_numpy()
                to_jitter = curiter_df[rows, cols]
                proposed_moves = self.rate_move * (
                    np.random.choice([1, -1], size=to_jitter.shape)
                )
                jittered = to_jitter + proposed_moves
                curiter_df[rows, cols] = jittered
                rc_df = rc_df.loc[rc_df["count"] > 1]
                rc_df.loc[:, "count"] = rc_df["count"] - 1
            curiter_df[curiter_df < 0.0] = 0.0
            curiter_df[curiter_df > self.rate_max] = self.rate_max
            self.add_to_result(curiter_df)

    def _run_rank_iter(self, _) -> list:
        """
        Single step of rank-jitter simulation. Swaps rank positions for random ballots.
        Returns computed IRV ranking.

        Returns:
            list: IRV ranking for the perturbed ballots.
        """
        cur_ballots_array = self.ballots_array.copy()
        sample_indices = self.rng.choice(
            len(self.ballots), size=int(self.sample_n / 2), replace=True
        )
        c_df = pd.DataFrame([sample_indices]).T.value_counts().reset_index(name="count")
        while c_df.shape[0] > 0:
            sample_indices = c_df.iloc[:, 0].to_numpy()
            col_1 = self.rng.choice(
                np.min([self.max_rank, self.df.shape[1]]), size=sample_indices.shape
            )
            col_2 = self.rng.choice([-1, 1], size=col_1.shape) + col_1
            col_2[col_2 < 0] = col_1[col_2 < 0]
            col_2[col_2 >= np.min([self.max_rank, self.df.shape[1]])] = col_1[
                col_2 >= np.min([self.max_rank, self.df.shape[1]])
            ]
            t = cur_ballots_array[sample_indices, col_1]
            cur_ballots_array[sample_indices, col_1] = cur_ballots_array[
                sample_indices, col_2
            ]
            cur_ballots_array[sample_indices, col_2] = t
            c_df = c_df.loc[c_df["count"] > 1]
            c_df.loc[:, "count"] = c_df["count"] - 1
        jittered_ballots = array_to_ballots(cur_ballots_array).tolist()
        irv_profile = PreferenceProfile(ballots=jittered_ballots)
        irv_election = IRV(profile=irv_profile)
        irv_rankings = irv_election.run_election().rankings()
        irv_places = []
        for ranking in irv_rankings:
            irv_places += list(ranking)
        return irv_places

    def _run_rank(self) -> None:
        """
        Jitter for 'rank' voting: swaps rank order for random ballots, computes IRV outcome.
        """
        self.rng = np.random.default_rng()
        self.ballots = np.array(
            generate_irv_ballots(
                pd.DataFrame(data=self.df, columns=self.labels), self.labels
            )
        )
        self.ballots_array = np.stack(ballots_to_array(self.ballots))
        for _ in np.arange(self.n_iters):
            self.ranking_iter_list.append(self._run_rank_iter(_))

    def _run_aa(self) -> None:
        """
        Jitter for 'accept_approve' voting: noise on categorical first-choice/acceptable/reject values.
        """
        rng = np.random.default_rng()
        for _ in range(self.n_iters):
            curiter_df = np.copy(self.df)
            curiter_df[np.isnan(curiter_df)] = 3
            rows = rng.choice(self.df.shape[0], size=self.sample_n)
            cols = rng.choice(self.df.shape[1], size=self.sample_n)
            rc_df = (
                pd.DataFrame([rows, cols]).T.value_counts().reset_index(name="count")
            )
            while rc_df.shape[0] > 0:
                rows = rc_df.iloc[:, 0].to_numpy()
                cols = rc_df.iloc[:, 1].to_numpy()
                to_jitter = curiter_df[rows, cols]
                proposed_moves = np.random.choice([1, -1], size=to_jitter.shape)
                jittered = to_jitter + proposed_moves
                jittered[jittered < 1] = 1
                jittered[jittered > 3] = 3
                curiter_df[rows, cols] = jittered
                rc_df = rc_df.loc[rc_df["count"] > 1]
                rc_df.loc[:, "count"] = rc_df["count"] - 1
            curiter_df[curiter_df == 3] = np.nan
            self.add_to_result_accept_approve(curiter_df)

    def _run_approval(self) -> None:
        """
        Jitter for 'approval' voting: flips approval status on sample entries.
        """
        rng = np.random.default_rng()
        for _ in range(self.n_iters):
            curiter_df = np.copy(self.df)
            rows = rng.choice(self.df.shape[0], size=self.sample_n)
            cols = rng.choice(self.df.shape[1], size=self.sample_n)
            rc_df = (
                pd.DataFrame([rows, cols]).T.value_counts().reset_index(name="count")
            )
            while rc_df.shape[0] > 0:
                rows = rc_df.iloc[:, 0].to_numpy()
                cols = rc_df.iloc[:, 1].to_numpy()
                to_jitter = curiter_df[rows, cols]
                to_jitter[to_jitter == 1] = 2
                to_jitter[np.isnan(to_jitter)] = 1
                to_jitter[to_jitter == 2] = 1
                curiter_df[rows, cols] = to_jitter
                rc_df = rc_df.loc[rc_df["count"] > 1]
                rc_df.loc[:, "count"] = rc_df["count"] - 1
            self.add_to_result(curiter_df)
