import random
from functools import partial
from typing import Callable, Literal

import numpy as np

from votekit.cleaning import remove_and_condense_rank_profile
from votekit.elections.election_state import ElectionState
from votekit.elections.election_types.ranking.abstract_ranking import RankingElection
from votekit.pref_profile import RankProfile
from votekit.utils import (
    first_place_votes,
    score_dict_to_ranking,
    tiebreak_set,
)


# TODO: make a parent class for these and have SerialVeto and PluralityVeto inherit from it
class PluralityVeto(RankingElection):
    """
    Scores each candidate by their plurality (number of first place) votes,
    then in a randomized order it lets each voter decrement the score of their
    least favorite remaining candidate. Candidates are eliminated when their score
    reaches 0. The last m candidates standing are the winners.

    Partial ballots are handled by assuming that unranked candidates tie for last place.

    Args:
        profile (RankProfile): RankProfile to run election on. Note that
            ballots must have integer weights to be considered valid for this mechanism.
        m (int): Number of seats to elect.
        tiebreak (str): Tiebreak method to use. Options are 'first_place', 'random', or 'borda'.
            Used to determine veto when multiple candidates are tied for last place on a ballot.
            Default is 'first_place'. If ``tiebreak`` is not 'random', tiebreak order is calculated
            in advance, using the initial profile. Thus, for 'borda' tiebreak, Borda scores are not
            recalculated as candidates are eliminated.
        scoring_tie_convention (str): How to award points for tied first-place votes. Defaults to
            "average", where if n candidates are tied for first, each receives 1/n points.
            "high" would award them each one point, and "low" 0.
            Used by ``score_function`` parameter.
            Also used to define ``tiebreak_order`` if tiebreak is 'first_place' or 'borda'.
        elimination_strategy (str): 'naive' eliminates candidates when their score hits zero.
            'careful' eliminates candidates only when they have zero score and a voter attempts to
            veto them (which does not use up that voter's veto).

    Attributes:
        m (int): The number of seats to be filled in the election.
        candidates (frozenset[str]): The set of candidates in the election.
        tiebreak_order (Optional[tuple[frozenset[str]]]): The candidate ordering used to break last-place ties
            when processing vetoes. ``None`` if ``tiebreak`` = 'random'.

    Raises:
        ValueError: If ``m`` is less than or equal to 0, or if ``m`` exceeds the number of
            candidates in the profile who received votes.
        TypeError: If the profile contains invalid ballots.
    """

    def __init__(
        self,
        profile: RankProfile,
        m: int = 1,
        tiebreak: Literal["first_place", "borda", "random", "lex"] = "first_place",
        scoring_tie_convention: Literal["high", "low", "average"] = "average",
        elimination_strategy: Literal["naive", "careful"] = "naive",
    ):
        grouped_profile = profile.group_ballots()

        self.m = m
        self.tiebreak = tiebreak
        self.scoring_tie_convention = scoring_tie_convention
        self._veto_loop: Callable[
            [dict[str, float]], tuple[frozenset[str], frozenset[str]]
        ] = (
            self._serial_veto_loop
            if elimination_strategy == "careful"
            else self._plurality_veto_loop
        )
        self._pv_validate_input(grouped_profile)

        self._df = grouped_profile.df.copy()
        self._ranking_cols = [
            f"Ranking_{i}" for i in range(1, grouped_profile.max_ranking_length + 1)
        ]
        self._rankings = self._df[self._ranking_cols].values
        # a ballot with weight w is a bin containing w voters
        # cumulative sum defines the boundaries of each bin, so we can index against it
        self._cumsum = self._df["Weight"].cumsum().to_numpy()

        num_voters = int(grouped_profile.total_ballot_wt)
        self._voter_order = np.random.permutation(num_voters)
        self._voter_order_pointer = 0

        self.candidates = frozenset(grouped_profile.candidates)
        self._eliminated = set("~")
        self._unlisted_candidates_by_ballot = tuple(
            self.candidates - (frozenset.union(*ballot[self._ranking_cols]))
            for _, ballot in self._df.iterrows()
        )

        # for each ballot, save the position where we last left off when looking for a veto
        self._veto_position_cache: list[int | None] = [None] * len(self._df)

        self.tiebreak_order = None
        if self.tiebreak != "random":
            # stores the most recent veto each ballot gave
            self._veto_cache = ["" for _ in range(len(self._df))]

            self.tiebreak_order = tiebreak_set(
                self.candidates,
                grouped_profile,
                self.tiebreak,
                scoring_tie_convention,
                backup_tiebreak_convention="lex",
            )
            self._tiebreak_ranks = {
                next(iter(s)): i for i, s in enumerate(self.tiebreak_order)
            }

        self._internal_round_number = 0

        super().__init__(
            grouped_profile,
            score_function=partial(
                first_place_votes, tie_convention=scoring_tie_convention
            ),
        )

    def _pv_validate_input(self, profile: RankProfile):
        """
        Validates input to PluralityVeto.
        Checks that each ballot has a ranking and that each ballot has integer weight.
        """
        if self.m <= 0:
            raise ValueError("m must be positive.")
        elif len(profile.candidates_cast) < self.m:
            raise ValueError("Not enough candidates received votes to be elected.")

        for ballot in profile.ballots:
            if ballot.ranking is None:
                raise TypeError("Ballots must have rankings.")
            elif int(ballot.weight) != ballot.weight:
                raise TypeError(f"Ballot {ballot} has non-integer weight.")

    def _get_ballot_idx(self, voter_idx: int) -> np.intp:
        """
        Converts a voter index in [0, num_voters) to a ballot index in [0, len(self._df)) that can
        be used to retrieve the row in self._df corresponding to that voter's ballot.
        """
        ballot_idx = np.searchsorted(self._cumsum, voter_idx, side="right")
        if ballot_idx >= len(self._df):
            raise IndexError(
                f"Voter index {voter_idx} out of range for a profile of {len(self._df)} ballots."
            )
        return ballot_idx

    def _break_tie(self, candidate_set: frozenset[str]) -> str:
        """
        Chooses a veto from a set of last-place candidates.
        If ``tiebreak`` = 'random', does so randomly. Otherwise, identifies veto according to
        ``tiebreak_order``, which is defined at instantiation.

        Args:
            candidate_set (frozenset[str]): The set of tied candidates.

        Returns:
            str: The candidate to be vetoed.
        """
        if self.tiebreak == "random":

            def rank(c: str) -> float:
                return random.random()

        # in _tiebreak_order, higher position is worse; veto the worst remaining
        else:

            def rank(c: str) -> float:
                return self._tiebreak_ranks[c]

        return max(candidate_set, key=rank)

    def _find_potential_vetoes_set(self, ballot_idx: np.intp) -> frozenset[str]:
        """
        Given a ballot index, returns the set of last-place candidates (before tiebreaking).
        First considers unlisted candidates; if all eliminated, walks backward through the
        ranking using the position cache.

        Args:
            ballot_idx (int): A ballot index in [0, len(self._df)).

        Returns:
            frozenset[str]: The candidate(s) tied for last place on this ballot.
        """
        cached_pos = self._veto_position_cache[ballot_idx]

        if cached_pos is None:
            potential_vetoes = (
                self._unlisted_candidates_by_ballot[ballot_idx] - self._eliminated
            )
            if potential_vetoes:
                return potential_vetoes
            # no unlisted candidates remain; start walking backwards from the end of the ranking
            cached_pos = len(self._ranking_cols) - 1

        ranking = self._rankings[ballot_idx]
        for pos in range(cached_pos, -1, -1):
            potential_vetoes = ranking[pos] - self._eliminated
            if potential_vetoes:
                self._veto_position_cache[ballot_idx] = pos
                break
        else:
            potential_vetoes = frozenset()

        return potential_vetoes

    def _get_veto(self, ballot_idx: np.intp) -> str:
        """
        Given a ballot index, returns the candidate to veto.
        For deterministic tiebreak methods, returns the most recent veto
        that ballot gave; if that candidate has been eliminated, calculates
        the new veto and updates _veto_cache.

        Args:
            ballot_idx (int): A ballot index in [0, len(self._df)).

        Returns:
            veto (str): The candidate to be vetoed.

        Raises:
            RuntimeError: If the ballot contains no remaining candidates.
        """
        if self.tiebreak != "random":
            most_recent_veto = self._veto_cache[ballot_idx]
            if most_recent_veto and most_recent_veto not in self._eliminated:
                return most_recent_veto

        potential_vetoes = self._find_potential_vetoes_set(ballot_idx)
        if not potential_vetoes:
            raise RuntimeError(
                "Attempted to get veto from a ballot that contained no remaining candidates."
            )
        veto = self._break_tie(potential_vetoes)

        if self.tiebreak != "random":
            self._veto_cache[ballot_idx] = veto
        return veto

    def _is_finished(self) -> bool:
        # for PluralityVeto, candidates are only elected in the final round, all at once
        elected = self.election_states[-1].elected
        elected = frozenset.union(*elected)
        return len(elected) > 0

    def _reset(self):
        """
        Resets _internal_round_number and _voter_order_pointer to 0, resets veto caches,
        and empties _eliminated so that the election can be replayed from the start.
        """
        self._internal_round_number = 0
        self._eliminated = set("~")
        self._voter_order_pointer = 0
        self._veto_position_cache = [None] * len(self._df)
        if self.tiebreak != "random":
            self._veto_cache = ["" for _ in range(len(self._df))]

    def _plurality_veto_loop(
        self, scores: dict[str, float]
    ) -> tuple[frozenset[str], frozenset[str]]:
        """
        Processes vetoes until some candidate's score reaches zero.
        Each voter decrements the score of their least favorite remaining candidate.

        Args:
            scores (dict[str, float]): Mutable score dict, modified in place.

        Returns:
            eliminated (frozenset[str]): Candidates worthy of elimination.
            elected (frozenset[str]): Candidates worthy of election.
        """

        eliminated = elected = ()
        if self._internal_round_number == 0:
            eliminated = tuple(c for c, score in scores.items() if score <= 0)

        while not eliminated:
            voter_idx = self._voter_order[self._voter_order_pointer]
            ballot_idx = self._get_ballot_idx(voter_idx)
            veto = self._get_veto(ballot_idx)

            scores[veto] -= 1
            self._voter_order_pointer += 1

            if scores[veto] <= 0:
                eliminated = (veto,)

        return frozenset(eliminated), frozenset(elected)

    def _serial_veto_loop(
        self, scores: dict[str, float]
    ) -> tuple[frozenset[str], frozenset[str]]:
        """
        Processes vetoes until some candidate is eliminated or all vetoes have been processed.
        Zero-score candidates are only eliminated when a voter attempts to veto them,
        which does not use up that voter's veto.
        If all vetoes are processed, elects all remaining candidates, breaking ties if necessary.

        Args:
            scores (dict[str, float]): Mutable score dict, modified in place.

        Returns:
            eliminated (frozenset[str]): Candidates worthy of elimination.
            elected (frozenset[str]): Candidates worthy of election.
        """
        eliminated = elected = ()
        while self._voter_order_pointer < len(self._voter_order):
            voter_idx = self._voter_order[self._voter_order_pointer]
            ballot_idx = self._get_ballot_idx(voter_idx)
            veto = self._get_veto(ballot_idx)

            if scores[veto] <= 0:
                eliminated = (veto,)
                break

            scores[veto] -= 1
            self._voter_order_pointer += 1
        else:
            # if we run out of voters, there's a tie
            elected = self.candidates - self._eliminated
        return frozenset(eliminated), frozenset(elected)

    def _run_step(
        self, profile: RankProfile, prev_state: ElectionState, store_states=False
    ) -> RankProfile:
        """
        Runs one round of the PluralityVeto election.
        If only ``m`` candidates remain, they are all elected.
        Otherwise, runs the veto loop to eliminate the next candidate.

        Args:
            profile (RankProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.
            store_states (bool, optional): True if ``self.election_states`` should be updated
                with the ElectionState generated by this round. This should only be True
                when used by ``self._run_election()``. Defaults to False.

        Returns:
            RankProfile: The profile of ballots after the round is completed,
            or an empty profile if the election has ended.
        """

        if self._internal_round_number != prev_state.round_number:
            if prev_state.round_number == 0:
                self._reset()
            else:
                raise ValueError(
                    "Calling _run_step on the middle of a PluralityVeto election is not permitted."
                )
        self._internal_round_number += 1

        new_scores = prev_state.scores.copy()
        remaining = self.candidates - self._eliminated
        if len(remaining) == self.m:
            electable_candidates = remaining
            eliminated = frozenset()
        else:
            eliminated, electable_candidates = self._veto_loop(new_scores)

        self._eliminated.update(eliminated)
        for c in eliminated:
            del new_scores[c]

        if electable_candidates:
            assert self.scoring_tie_convention in ("high", "average", "low")
            tiebroken_order = tiebreak_set(
                electable_candidates,
                profile,
                self.tiebreak,
                self.scoring_tie_convention,
                backup_tiebreak_convention="lex",
            )
            elected = tiebroken_order[: self.m]
            new_profile = RankProfile()
        else:
            elected = (frozenset(),)
            new_profile = remove_and_condense_rank_profile(
                removed=list(eliminated),
                profile=profile,
                remove_zero_weight_ballots=False,
                remove_empty_ballots=False,
            )

        if store_states:
            remaining = score_dict_to_ranking(new_scores)
            new_state = ElectionState(
                round_number=prev_state.round_number + 1,
                elected=elected,
                eliminated=(eliminated,),
                remaining=remaining,
                scores=new_scores,
            )

            self.election_states.append(new_state)
        return new_profile
