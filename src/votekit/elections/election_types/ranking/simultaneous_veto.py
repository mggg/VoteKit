from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np
from typing_extensions import Sentinel

from votekit.cleaning import remove_and_condense_rank_profile
from votekit.elections.election_state import ElectionState
from votekit.elections.election_types.ranking.abstract_ranking import RankingElection
from votekit.pref_profile import RankProfile
from votekit.utils import (
    borda_scores,
    first_place_votes,
    score_dict_from_score_vector,
    score_dict_to_ranking,
    tiebreak_set,
)

NO_CANDIDATES_REMAINING = Sentinel("NO_CANDIDATES_REMAINING")
NO_POSITION = -1


@dataclass(frozen=True, slots=True)
class _SVState:
    veto_matrix: np.ndarray
    veto_position_cache: np.ndarray
    scores: np.ndarray
    eliminated: frozenset[str]


class SimultaneousVeto(RankingElection):
    """
    An anonymous version of PluralityVeto.

    Each candidate begins with a score, then, all at the same time, voters remove score from their
    least favorite candidates at a constant rate. As candidates' scores reach zero, they are
    eliminated, and voters that were vetoing them transfer to their next least-favorite candidate.
    With plurality scores (first-place votes), this voting rule achieves optimal metric distortion.
    See Kizilkaya & Kempe's 2023 paper: "Generalized Veto Core and a Practical Voting Rule with
    Optimal Metric Distortion" (https://arxiv.org/pdf/2305.19632).

    Partial ballots are treated by assuming that all unranked candidates tie for last place.
    Voters with ties in their ranking will spread their veto evenly across the tied candidates.

    Args:
        profile (RankProfile): Profile to run election on.
        m (int, optional): Number of seats to elect. Defaults to 1.
        candidate_weights (Literal['first_place', 'uniform', 'borda', 'harmonic'] | dict[str, float]
            | int, optional): Initial candidate scores. 'first_place' means candidates begin with
            their first-place vote count. 'uniform' means all candidates begin with the same
            score. 'borda' means candidates begin with their Borda scores. If a dictionary,
            keys are candidates and values are initial scores; a score must be provided
            for every candidate. If an integer k, candidates begin with their top-k vote count.
            Defaults to "first_place".
        tiebreak (Literal['first_place', 'random', 'borda', 'remaining_score', 'veto_pressure',
            'lex'], optional): Method for breaking ties when multiple candidates
            are eliminated simultaneously. Defaults to "first_place".
            Backup tiebreak is lexicographic/alphabetical.
        scoring_tie_convention (Literal['high', 'average', 'low'], optional): How to
            award points for ties in rankings when ``candidate_weights`` is 'first_place'
            or 'borda'. Defaults to "average".
        return_all_tied_winners (bool): If True, election returns a winner set of all tied winners,
            even if it is larger than ``m``. Defaults to False.

    Attributes:
        candidates (frozenset[str]): Candidates in the initial profile.
        initial_scores (dict[str, float]): Initial scores of each candidate before veto process.

    Raises:
        ValueError: If any of the following:
            - ``m`` is not positive or exceeds the number of candidates.
            - any ballot lacks a ranking.
            - ``candidate_weights`` is a dict missing candidates as keys.
            - ``candidate_weights`` is an unrecognized string.
            - ``candidate_weights`` is an int not in [1, number of candidates]
        TypeError: If ``candidate_weights`` is not a string, dict, or int.
    """

    VALID_TIEBREAKS = {
        "first_place",
        "random",
        "borda",
        "remaining_score",
        "veto_pressure",
        "lex",
    }
    VALID_TIE_CONVENTIONS = {"high", "average", "low"}

    def __init__(
        self,
        profile: RankProfile,
        m: int = 1,
        candidate_weights: (
            Literal["first_place", "uniform", "borda", "harmonic"] | dict[str, float] | int
        ) = "first_place",
        tiebreak: Literal[
            "first_place", "random", "borda", "remaining_score", "veto_pressure", "lex"
        ] = "first_place",
        scoring_tie_convention: Literal["high", "average", "low"] = "average",
        return_all_tied_winners: bool = False,
    ):
        self._sv_validate_input(profile, m, tiebreak, scoring_tie_convention)
        self.m = m
        self.candidate_weights = candidate_weights
        self.tiebreak = tiebreak
        self.scoring_tie_convention = scoring_tie_convention
        self.return_all_tied_winners = return_all_tied_winners

        grouped_profile = profile.group_ballots()
        self._df = grouped_profile.df.copy()
        self.candidates = frozenset(grouped_profile.candidates_cast)
        self._eliminated: set[str] = set("~")

        self._sorted_candidates = tuple(sorted(self.candidates))
        self._candidate_to_idx = {c: i for i, c in enumerate(self._sorted_candidates)}

        # unmentioned candidates are considered tied for last place
        assert grouped_profile.max_ranking_length is not None
        self._max_ranking_length: int = grouped_profile.max_ranking_length
        self._ranking_cols = [f"Ranking_{i}" for i in range(1, self._max_ranking_length + 1)]
        self._unlisted_candidates_by_ballot = tuple(
            self.candidates - (frozenset.union(*ballot[self._ranking_cols]))
            for _, ballot in self._df.iterrows()
        )

        n_candidates = len(self.candidates)
        n_ballots = len(self._df)

        # scores are stored as a vector where position indicates the candidate
        self._scores = np.zeros(n_candidates)
        score_func = self._make_score_function(candidate_weights)
        for candidate, score in score_func(grouped_profile).items():
            candidate_idx = self._candidate_to_idx[candidate]
            self._scores[candidate_idx] = score
        self.initial_scores = self._compute_scores_dict()

        # for each ballot, save the position where we last left off when looking for a veto
        self._veto_position_cache = np.full(shape=(n_ballots,), fill_value=NO_POSITION)

        self._veto_matrix = self._initialize_veto_matrix(n_candidates, n_ballots)

        self._sv_states: dict[int, _SVState] = {}

        super().__init__(grouped_profile, score_function=score_func)

    def _compute_scores_dict(self) -> dict[str, float]:
        """
        Converts self._scores (np.array) to dict[str, float].

        Returns:
            dict[str, float]: Dictionary mapping candidates to scores.
        """
        return {
            cand: self._scores[idx]
            for idx, cand in enumerate(self._sorted_candidates)
            if cand not in self._eliminated
        }

    def _sv_validate_input(
        self,
        profile: RankProfile,
        m: int,
        tiebreak: str,
        scoring_tie_convention: str,
    ):
        """
        Validates input to SimultaneousVeto.

        Checks that each ballot has a ranking, that ``m`` is valid,
        that enough candidates received votes to fill all the seats,
        and that if candidate_weights is an int, it's valid.

        Args:
            profile (RankProfile): RankProfile to run election on.
            m (int, optional): Number of seats to elect. Defaults to 1.
            tiebreak (str, optional): Method for breaking ties when multiple candidates
                are eliminated simultaneously. Defaults to "first_place".
                Backup tiebreak is lexicographic/alphabetical.
            scoring_tie_convention (str, optional): How to award points for ties in rankings
                when ``candidate_weights`` is 'first_place' or 'borda'. Defaults to "average".

        Raises:
            ValueError: If ``m`` is not positive or exceeds the number of
                candidates, any ballot lacks a ranking, tiebreak is not valid,
                or scoring_tie_convention is not valid.
        """
        if m <= 0:
            raise ValueError("m must be positive.")
        if len(profile.candidates_cast) < m:
            raise ValueError("Not enough candidates received votes to be elected.")
        if any(ballot.ranking is None for ballot in profile.ballots):
            raise ValueError("Ballots must have rankings.")
        if tiebreak not in self.VALID_TIEBREAKS:
            raise ValueError(
                f"tiebreak={tiebreak} is not valid. Did you mean one of {self.VALID_TIEBREAKS}?"
            )
        if scoring_tie_convention not in self.VALID_TIE_CONVENTIONS:
            raise ValueError(
                f"tiebreak={scoring_tie_convention} is not valid. "
                f"Did you mean one of {self.VALID_TIE_CONVENTIONS}?"
            )

    def _make_score_function(
        self, candidate_weights: str | dict[str, float] | int
    ) -> Callable[[RankProfile], dict[str, float]]:
        """
        Converts ``candidate_weights`` into a callable function.

        This function is used to generate initial scores and is also passed to super().__init__.

        Args:
            candidate_weights (str | dict[str, float] | int):
                How to initialize candidate scores. 'first_place' means candidates begin with their
                first-place vote count. 'uniform' means all candidates begin with the same
                score. 'borda' means candidates begin with their Borda scores. If a dictionary,
                keys are candidates and values are initial scores; a score must be provided
                for every candidate. If an integer k, candidates begin with their top-k vote count.

        Returns:
            Callable[[RankProfile], dict[str, float]]: Score function that takes a RankProfile and
                returns a dict mapping candidates to scores.

        Raises:
            ValueError: If any of the following:
                - ``candidate_weights`` is an int not in [1, number of candidates].
                - ``candidate_weights`` is a dict missing candidates as keys.
                - ``candidate_weights`` is an unrecognized string.
            TypeError: If ``candidate_weights`` is not a string or dict.
        """
        if isinstance(k := candidate_weights, int):
            if k < 1:
                raise ValueError(f"Invalid value for top-k scoring: candidate_weights={k}")
            if k > self._max_ranking_length:
                raise ValueError(
                    f"candidate_weights={k} is not valid for a profile with "
                    f"max_ranking_length={self._max_ranking_length}."
                )
            assert self.scoring_tie_convention in ("high", "average", "low")
            return partial(
                score_dict_from_score_vector,
                score_vector=[1] * k,
                tie_convention=self.scoring_tie_convention,
            )
        match candidate_weights:
            case "first_place" | "plurality":
                assert self.scoring_tie_convention in ("high", "average", "low")
                return partial(first_place_votes, tie_convention=self.scoring_tie_convention)
            case "borda":
                assert self.scoring_tie_convention in ("high", "average", "low")
                return partial(borda_scores, tie_convention=self.scoring_tie_convention)
            case "uniform":

                def uniform_weights(profile: RankProfile) -> dict[str, float]:
                    return {c: 1.0 for c in profile.candidates}

                return uniform_weights
            case "harmonic" | "dowdall":

                def harmonic_weights(profile: RankProfile) -> dict[str, float]:
                    assert profile.max_ranking_length is not None
                    harmonic_score_vector = [1 / (i + 1) for i in range(profile.max_ranking_length)]
                    return score_dict_from_score_vector(
                        profile, harmonic_score_vector, self.scoring_tie_convention
                    )

                return harmonic_weights
            case _:
                if isinstance(candidate_weights, str):
                    raise ValueError(
                        f"Received invalid input for candidate weights: {candidate_weights}"
                    )
                if not isinstance(candidate_weights, Mapping):
                    raise TypeError(
                        "Exected for 'candidate_weights' to be either a string or a dictionary mapping "
                        f"candidate names to their weights. Found {type(candidate_weights)!r}"
                    )
                missing_cands = self.candidates.difference(candidate_weights.keys())
                if missing_cands:
                    msg = (
                        "If candidate_weights is a dict, "
                        "scores must be provided for all candidates. "
                        f"The following candidates were missing: {missing_cands}"
                    )
                    raise ValueError(msg)

                def custom_weights(profile: RankProfile) -> dict[str, float]:
                    return {c: candidate_weights[c] for c in profile.candidates}

                return custom_weights

    def _initialize_veto_matrix(self, n_candidates: int, n_ballots: int) -> np.ndarray:
        """
        Initializes veto_matrix.

        Args:
            n_candidates (int): Number of candidates.
            n_ballots (int): Number of ballots.

        Returns:
            np.ndarray: A veto matrix of shape (n_candidates, n_ballots), where element (i,j)
                represents the amount of veto ballot j is giving to candidate i.
        """
        veto_matrix = np.zeros((n_candidates, n_ballots))
        for ballot_idx in np.arange(n_ballots, dtype=np.intp):
            veto_weight = self._df["Weight"].iloc[ballot_idx]
            vetoes = self._get_vetoes(ballot_idx)
            veto_indices = [self._candidate_to_idx[c] for c in vetoes]
            veto_weight /= len(veto_indices)
            veto_matrix[veto_indices, ballot_idx] = veto_weight
        return veto_matrix

    def _update_veto_matrix(self, candidate: str):
        """
        Updates veto matrix in place by redistributing veto pressure from an eliminated candidate.

        Args:
            candidate (str): Candidate being eliminated.
        """
        candidate_idx = self._candidate_to_idx[candidate]
        ballots_to_update = np.flatnonzero(self._veto_matrix[candidate_idx])
        for ballot_idx in ballots_to_update:
            veto_weight = self._veto_matrix[candidate_idx, ballot_idx]
            self._veto_matrix[candidate_idx, ballot_idx] = 0
            # if some other candidates were already being vetoed,
            # they were tied for last place on this ballot
            veto_indices = np.flatnonzero(self._veto_matrix[:, ballot_idx])
            if veto_indices.size == 0:
                vetoes = self._get_vetoes(ballot_idx)
                veto_indices = np.array([self._candidate_to_idx[c] for c in vetoes])
            veto_weight /= len(veto_indices)
            self._veto_matrix[veto_indices, ballot_idx] += veto_weight

    def _get_vetoes(self, ballot_idx: np.intp) -> frozenset[str]:
        """
        Given a ballot index, returns the candidate(s) to veto.

        When calculating veto, first considers unlisted candidates.
        If they have all been eliminated, then it walks backward through the ballot
        until it finds a set with remaining candidate(s).

        Uses ``_veto_position_cache`` to remember where each ballot was last vetoing,
        so we don't have to walk from the end of the ranking every time.

        Args:
            ballot_idx (np.intp): A ballot index in [0, len(self._df)) corresponds to the row in
                self._df containing that ballot.

        Returns:
            frozenset[str]: The candidate(s) to be vetoed.

        Raises:
            ValueError: If the ballot has no remaining candidates to veto.
        """
        cached_pos = self._veto_position_cache[ballot_idx]

        if cached_pos == NO_POSITION:
            vetoes = self._unlisted_candidates_by_ballot[ballot_idx] - self._eliminated
            if vetoes:
                return vetoes
            cached_pos = self._max_ranking_length - 1

        ballot = self._df.iloc[ballot_idx]
        ranking = ballot[self._ranking_cols].values

        for pos in range(cached_pos, -1, -1):
            vetoes = ranking[pos] - self._eliminated
            if vetoes:
                self._veto_position_cache[ballot_idx] = pos - 1
                break
        else:
            raise RuntimeError(
                "Attempted to get veto from a ballot that contained no remaining candidates."
            )
        return vetoes

    def _is_finished(self) -> bool:
        """Returns True if election is finished, False if another round is needed."""
        # for SimultaneousVeto, candidates are only elected in the final round, all at once
        elected = self.election_states[-1].elected
        elected_set = frozenset.union(*elected)
        return len(elected_set) > 0

    def _break_tie(
        self,
        candidates: frozenset[str],
        candidate_idx: Iterable[int],
        profile: RankProfile,
    ) -> tuple[frozenset[str], ...]:
        """
        Takes candidate names and indices and returns a tiebroken order of names.

        Args:
            candidates (frozenset[str]): Names of tied candidates.
            candidate_idx (Iterable[int]): Indices of tied candidates.
            profile (RankProfile): RankProfile of the current round.
                Passed to tiebreak_set() if ``tiebreak`` is not 'veto_pressure'
                or 'remaining_score'.

        Returns:
            tuple[frozenset[str], ...]: Tiebroken ordering of candidates (each in their own set).
        """

        def make_singleton_ranking(indices: list[int]) -> tuple[frozenset[str], ...]:
            """Convert sorted candidate indices to a tuple of singleton frozensets."""
            return tuple(frozenset((self._sorted_candidates[i],)) for i in indices)

        match self.tiebreak:
            case "veto_pressure":
                # sort eliminatable candidates in order of increasing veto pressure
                sorted_indices = sorted(candidate_idx, key=lambda i: self._veto_pressure[i])
                tiebroken_order = make_singleton_ranking(sorted_indices)
            case "remaining_score":
                # sort eliminatable candidates in order of decreasing remaining score prior to step
                sorted_indices = sorted(
                    candidate_idx, key=lambda i: self._veto_pressure[i], reverse=True
                )
                tiebroken_order = make_singleton_ranking(sorted_indices)
            case _:
                assert self.scoring_tie_convention in ("high", "low", "average")
                tiebroken_order = tiebreak_set(
                    candidates,
                    profile,
                    tiebreak=self.tiebreak,
                    scoring_tie_convention=self.scoring_tie_convention,
                )

        return tiebroken_order

    def _eliminate_one_candidate(
        self, profile: RankProfile
    ) -> tuple[str | None, dict[frozenset[str], tuple[frozenset[str], ...]]]:
        """
        Eliminate exactly one candidate whose score has hit zero, breaking a tie if necessary.

        Args:
            profile (RankProfile): RankProfile of the current round.

        Returns:
            tuple[str | None, dict[frozenset[str], tuple[frozenset[str], ...]]]:
                Returns a tuple (eliminated_candidate, tiebreaks), where eliminated_candidate
                is either a str giving the name of the eliminated candidate, or ``None``,
                signaling that no candidate was eliminated; and tiebreaks is a dict
                mapping a set of simultaneously-eliminated candidates to a tiebroken order;
                if only one candidate is eliminated, tiebreaks is empty.
        """
        idx_to_elim = np.where((self._scores <= 0) & (self._veto_pressure > 0))[0]

        tiebreaks: dict[frozenset[str], tuple[frozenset[str], ...]] = {}
        match idx_to_elim.size:
            case 0:
                return None, tiebreaks
            case 1:
                eliminated_candidate = self._sorted_candidates[idx_to_elim[0]]
            case _:
                candidates_to_elim = frozenset(self._sorted_candidates[idx] for idx in idx_to_elim)
                tiebroken_order = self._break_tie(
                    candidates=candidates_to_elim,
                    candidate_idx=idx_to_elim,
                    profile=profile,
                )
                eliminated_candidate = next(iter(tiebroken_order[-1]))
                tiebreaks = {candidates_to_elim: tiebroken_order}

        self._eliminated.add(eliminated_candidate)
        self._update_veto_matrix(eliminated_candidate)
        return eliminated_candidate, tiebreaks

    def _handle_all_zeroed(
        self, profile: RankProfile
    ) -> tuple[Sentinel, dict[frozenset[str], tuple[frozenset[str], ...]]]:
        """
        Handles the case in which all remaining candidates' scores hit zero simultaneously.

        This represents the end of the election, and the remaining candidates are tied winners.
        Because there may be more than ``m`` candidate remaining, prepares a tiebroken order.

        Args:
            profile (RankProfile): RankProfile of the current round.

        Returns:
            tuple[Sentinel, dict[frozenset[str], tuple[frozenset[str], ...]]]:
                Returns a tuple (eliminated_candidate, tiebreaks), where eliminated_candidate
                is a Sentinel indicating that the election is over, and tiebreaks is a dict
                mapping the set of remaining candidates to a tiebroken order of the same.
        """
        tiebreaks = {}
        if not self.return_all_tied_winners:
            remaining = self.candidates - self._eliminated
            candidate_idx = np.array([self._candidate_to_idx[c] for c in remaining])
            tiebroken_order = self._break_tie(
                candidates=remaining, candidate_idx=candidate_idx, profile=profile
            )
            tiebreaks = {remaining: tiebroken_order}
        return NO_CANDIDATES_REMAINING, tiebreaks

    def _veto_step(
        self, profile: RankProfile
    ) -> tuple[str | Sentinel | None, dict[frozenset[str], tuple[frozenset[str], ...]]]:
        """
        Core of the SimultaneousVeto algorithm.

        Calculates the amount of veto_pressure on the remaining candidates.
        Calculates delta, the minimum step forward required to reduce
        at least one candidate's score to zero.
        Decrements each candidate's score by delta * veto_pressure.
        If multiple candidates' scores are reduced to zero, breaks the tie
        and eliminates exactly one of them.

        If all candidates scores are reduced to zero, the election is over,
        and we break the tie between the remaining candidates.

        Args:
            profile (RankProfile): Profile from the current round of the election.
                Used for tiebreaking, if necessary.

        Returns:
            tuple[str | Sentinel | None, dict[frozenset[str], tuple[frozenset[str], ...]]]:
                A 2-tuple of (eliminated_candidate, tiebreaks). eliminated_candidate is one of:
                    - a str indicating the candidate to be eliminated
                    - NO_CANDIDATES_REMAINING, a Sentinel indicating the end of the election
                    - None, an error code signaling the failure to eliminate a candidate this round
                and tiebreaks is a dict mapping an unordered frozenset of candidates to their
                tiebroken order (a tuple of singleton frozensets).
        """

        self._veto_pressure = self._veto_matrix.sum(axis=1)
        mask = self._veto_pressure != 0
        delta = np.min(self._scores[mask] / self._veto_pressure[mask])
        self._scores -= delta * self._veto_pressure
        # handle floating point imprecision:
        self._scores[np.abs(self._scores) < 1e-10] = 0

        eliminated_candidate: str | Sentinel | None = None
        if np.any(self._scores):
            eliminated_candidate, tiebreaks = self._eliminate_one_candidate(profile)
        else:
            eliminated_candidate, tiebreaks = self._handle_all_zeroed(profile)
        return eliminated_candidate, tiebreaks

    def _write_state(self, round_number: int):
        """
        Saves the internal election state for the start of this round.

        Args:
            round_number (int): Round number for the current round.
                Used as the key for writing the current state to ``_sv_states``.
        """
        self._sv_states[round_number] = _SVState(
            veto_matrix=np.array(self._veto_matrix),
            veto_position_cache=np.array(self._veto_position_cache),
            scores=np.array(self._scores),
            eliminated=frozenset(self._eliminated),
        )

    def _load_state(self, round_number: int):
        """
        If the election has already been run, loads the state for the start of this round.

        Args:
            round_number (int): Round number for the current round.
                Used as the key a lookup in ``_sv_states``.
        """
        state = self._sv_states[round_number]
        self._veto_matrix = np.array(state.veto_matrix)
        self._veto_position_cache = np.array(state.veto_position_cache)
        self._scores = np.array(state.scores)
        self._eliminated = set(state.eliminated)

    def _run_step(
        self, profile: RankProfile, prev_state: ElectionState, store_states=False
    ) -> RankProfile:
        """
        Runs one round of a SimultaneousVeto election.

        If exactly m candidates remain, they are elected.
        Otherwise, runs the veto step, which either eliminates a single candidate
        or reduces all candidates' scores to zero, ending the election.

        Args:
            profile (RankProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.
            store_states (bool, optional): True if ``self.election_states`` should be updated
                with the ElectionState generated by this round. This should only be True when used
                by ``self._run_election()``. Defaults to False.

        Returns:
            RankProfile: The profile of ballots after the round is completed,
                or an empty profile if the election has ended.
        """

        current_round = prev_state.round_number + 1
        if current_round in self._sv_states:
            self._load_state(current_round)
        else:
            self._write_state(current_round)

        eliminated: tuple[frozenset[str], ...] = (frozenset(),)
        elected: tuple[frozenset[str], ...] = (frozenset(),)
        tiebreaks: dict[frozenset[str], tuple[frozenset[str], ...]] = {}

        remaining_set = self.candidates - self._eliminated
        if len(remaining_set) <= self.m:
            new_profile = RankProfile()
            elected = prev_state.remaining
        else:
            eliminated_candidate, tiebreaks = self._veto_step(profile)

            if eliminated_candidate is None:
                raise RuntimeError(
                    f"Round {prev_state.round_number} failed to eliminate any candidates."
                )

            if eliminated_candidate is NO_CANDIDATES_REMAINING:
                new_profile = RankProfile()
                if self.return_all_tied_winners:
                    elected = (remaining_set,)
                else:
                    elected = tiebreaks[remaining_set][: self.m]  # elect top m
            else:
                assert isinstance(eliminated_candidate, str)
                eliminated = (frozenset((eliminated_candidate,)),)
                new_profile = remove_and_condense_rank_profile(
                    removed=eliminated_candidate,
                    profile=profile,
                    remove_zero_weight_ballots=False,
                    remove_empty_ballots=False,
                )

        if store_states:
            scores = self._compute_scores_dict()
            remaining = score_dict_to_ranking(scores)
            remaining = tuple(c for c in remaining if c not in elected)
            new_state = ElectionState(
                round_number=current_round,
                eliminated=eliminated,
                elected=elected,
                remaining=remaining,
                scores=scores,
                tiebreaks=tiebreaks,
            )
            self.election_states.append(new_state)

        return new_profile
