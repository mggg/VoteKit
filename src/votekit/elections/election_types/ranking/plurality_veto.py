import random
from collections import deque
from functools import partial
from typing import Literal

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
        self.m = m
        self.tiebreak = tiebreak
        self.scoring_tie_convention = scoring_tie_convention
        self.elimination_strategy = elimination_strategy
        self._pv_validate_input(profile)

        # --- dataframe setup ---
        profile = profile.group_ballots()
        self._df = profile.df.copy()
        self._ranking_cols = [
            f"Ranking_{i}" for i in range(1, profile.max_ranking_length + 1)
        ]
        self._rankings = self._df[self._ranking_cols].values
        # a ballot with weight w is a bin containing w voters
        # using cumulative sum defines the boundaries of each bin, so we can index against it
        self._cumsum = self._df["Weight"].cumsum().to_numpy()

        # --- voter order ---
        # (save the initial order so it is replayable)
        num_voters = int(profile.total_ballot_wt)
        self._initial_voter_order = np.random.permutation(num_voters)
        self._voter_order = deque(self._initial_voter_order)

        # --- candidate tracking ---
        self.candidates = frozenset(profile.candidates)

        # '~' is the symbol for a ballot position where no candidates were ranked
        # so we discard it wherever we see it
        self._eliminated = set("~")

        # calculate the candidates not mentioned for each ballot
        # we consider them to be tied for last place
        self._unlisted_candidates = tuple(
            self.candidates - (frozenset.union(*ballot[self._ranking_cols]))
            for _, ballot in self._df.iterrows()
        )

        # for each ballot, we save the position where we last left off when looking for a veto
        # so we don't have to walk backwards from the end of the ballot every time
        self._veto_position_cache: list[int | None] = [None] * len(self._df)

        # --- tiebreaking setup ---
        self.tiebreak_order = None
        if self.tiebreak != "random":
            # since we may access a ballot many times in a round,
            # we cache the most recent veto that ballot gave
            # to reduce calls to _break_tie.
            # unused if tiebreak = 'random'
            self._veto_cache = ["" for _ in range(len(self._df))]

            # for deterministic tiebreaks, we calculate the tiebreak order in advance
            self.tiebreak_order = tiebreak_set(
                self.candidates,
                profile,
                self.tiebreak,
                self.scoring_tie_convention,
                backup_tiebreak_convention="lex",  # lexicographic/alphabetical tiebreaking
            )
            # we use a dict version of the same for quick lookups
            self._tiebreak_order = {
                next(iter(s)): i for i, s in enumerate(self.tiebreak_order)
            }

        # internal round number is used to check if _run_step is being called after the election
        # was already run (e.g., by get_profile())
        self._internal_round_number = 0

        super().__init__(
            profile,
            score_function=partial(
                first_place_votes, tie_convention=scoring_tie_convention
            ),
        )

    def _pv_validate_input(self, profile: RankProfile):
        """
        Validates input to PluralityVeto.
        Checks that each ballot has a ranking and that each
        ballot has integer weight.
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

        Raises:
            ValueError: If candidate_set is empty, or if candidate_set does not contain any of the
            remaining candidates.
        """
        if not candidate_set:
            raise ValueError("Cannot tie-break an empty set.")
        if not candidate_set.intersection(self.candidates - self._eliminated):
            raise ValueError(
                f"Tried to tie-break {candidate_set}, but it contains no remaining candidates."
            )
        if len(candidate_set) == 1:
            return next(iter(candidate_set))
        if self.tiebreak == "random":
            return random.choice(list(candidate_set))
        assert self._tiebreak_order

        # in _tiebreak_order, higher position is worse; veto the worst remaining
        def rank(c: str) -> int:
            return self._tiebreak_order[c]

        return max(candidate_set, key=rank)

    def _find_vetoes(self, ballot_idx: np.intp) -> frozenset[str]:
        """
        Given a ballot index, returns the set of last-place candidates (before tiebreaking).
        First considers unlisted candidates; if all eliminated, walks backward through the
        ranking using the position cache.

        Args:
            ballot_idx (int): A ballot index in [0, len(self._df)).

        Returns:
            frozenset[str]: The candidate(s) tied for last place on this ballot.

        Raises:
            ValueError: If the ballot has no remaining candidates.
        """
        cached_pos = self._veto_position_cache[ballot_idx]

        if cached_pos is None:
            potential_vetoes = self._unlisted_candidates[ballot_idx] - self._eliminated
            if potential_vetoes:
                return potential_vetoes
            # no unlisted candidates remain; start walking backwards from the end of the ranking
            cached_pos = len(self._ranking_cols) - 1

        ranking = self._rankings[ballot_idx]
        for pos in range(cached_pos, -1, -1):
            potential_vetoes = ranking[pos] - self._eliminated
            if potential_vetoes:
                self._veto_position_cache[ballot_idx] = pos
                return potential_vetoes

        raise ValueError(
            f"Ballot {ranking} was depleted before identifying a remaining candidate."
        )

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
        """
        if self.tiebreak != "random":
            most_recent_veto = self._veto_cache[ballot_idx]
            if most_recent_veto and most_recent_veto not in self._eliminated:
                return most_recent_veto

        potential_vetoes = self._find_vetoes(ballot_idx)
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
        """Reset all mutable election state so the election can be replayed from round 0."""
        self._internal_round_number = 0
        self._eliminated = set("~")
        self._voter_order = deque(self._initial_voter_order)
        self._veto_position_cache = [None] * len(self._df)
        if self.tiebreak != "random":
            assert self.tiebreak_order
            self._veto_cache = ["" for _ in range(len(self._df))]

    def _plurality_veto_step(self, scores):
        """
        Processes voters until some candidate's score reaches zero.
        Each voter decrements the score of their least favorite remaining candidate.
        The candidate whose score hits zero is returned.

        Args:
            scores (dict[str, float]): Mutable score dict, modified in place.

        Returns:
            str: The eliminated candidate.
        """
        while True:
            voter_idx = self._voter_order.pop()
            ballot_idx = self._get_ballot_idx(voter_idx)
            veto = self._get_veto(ballot_idx)

            # deduct one point for each veto
            scores[veto] -= 1

            # eliminate any candidate whose score reaches zero
            if scores[veto] <= 0:
                return veto

    def _serial_veto_step(self, scores):
        """
        Processes voters until a zero-score candidate is targeted.
        When a voter's veto target already has score <= 0, that candidate is eliminated
        and the voter's veto is not consumed (they are returned to the deque).
        Otherwise, the voter decrements their target's score normally.

        Args:
            scores (dict[str, float]): Mutable score dict, modified in place.

        Returns:
            str | None: The eliminated candidate, or ``None`` if all voters are
            exhausted without an elimination (indicating a tied winner set).
        """
        while self._voter_order:
            voter_idx = self._voter_order.pop()
            ballot_idx = self._get_ballot_idx(voter_idx)
            veto = self._get_veto(ballot_idx)

            # eliminate an opposed candidate whose score reaches zero and is opposed by this voter
            if scores[veto] <= 0:
                # return this voter to the deque since we didn't use their veto
                self._voter_order.append(voter_idx)
                return veto

            # otherwise deduct one point from the vetoed candidate
            scores[veto] -= 1

        # if we run out of voters, it means there's a tie
        return None

    def _elect_remaining(self, prev_state, store_states):
        """
        Elects all remaining candidates, ending the election.
        Used when only ``m`` candidates remain, or when the careful strategy
        exhausts all voters before eliminating enough candidates.

        Args:
            prev_state (ElectionState): The previous ElectionState.
            store_states (bool): Whether to append the final ElectionState.

        Returns:
            RankProfile: An empty profile representing the end of the election.
        """
        elected = prev_state.remaining
        new_profile = RankProfile()

        if store_states:
            assert self.score_function
            scores = self.score_function(new_profile)
            new_state = ElectionState(
                round_number=prev_state.round_number + 1,
                elected=elected,
                scores=scores,
            )

            self.election_states.append(new_state)

        return new_profile

    def _run_step(
        self, profile: RankProfile, prev_state: ElectionState, store_states=False
    ) -> RankProfile:
        """
        Runs one round of the PluralityVeto election.
        If only ``m`` candidates remain, they are all elected.

        Behavior depends on ``elimination_strategy``:

        - **naive**: On round 0, eliminates any candidates with zero first-place votes.
          If none were eliminated (or on later rounds), processes voters one at a time:
          each voter decrements the score of their least favorite remaining candidate,
          and the first candidate whose score reaches zero is eliminated.
        - **careful**: Processes voters one at a time. A candidate is only eliminated
          when a voter targets them and their score is already zero; that voter's veto
          is not consumed. If all voters are exhausted before an elimination, the
          remaining candidates are elected as a tied winner set.

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

        # if _run_step is called with an unexpected round number,
        # that round number must be 0, in which case we re-initialize the election
        if self._internal_round_number != prev_state.round_number:
            if prev_state.round_number == 0:
                self._reset()
            else:
                raise ValueError(
                    "Calling _run_step on the middle of a PluralityVeto election is not permitted."
                )

        self._internal_round_number += 1

        remaining = self.candidates - self._eliminated
        if len(remaining) == self.m:
            return self._elect_remaining(prev_state, store_states=store_states)

        new_scores = prev_state.scores.copy()

        eliminated_this_round = []
        if self.elimination_strategy == "naive":
            # eliminate candidates that received no first-place votes
            if prev_state.round_number == 0:
                for c, score in new_scores.items():
                    if score <= 0:
                        eliminated_this_round.append(c)

            # skip veto loop if we already eliminated zero-score candidate
            if not eliminated_this_round:
                eliminated_candidate = self._plurality_veto_step(new_scores)
                eliminated_this_round.append(eliminated_candidate)

        elif self.elimination_strategy == "careful":
            # SerialVeto only eliminates candidates when they have zero score and some voter
            # attempts to veto them
            eliminated_candidate = self._serial_veto_step(new_scores)
            if eliminated_candidate is None:
                return self._elect_remaining(prev_state, store_states=store_states)

            eliminated_this_round.append(eliminated_candidate)

        for c in eliminated_this_round:
            self._eliminated.add(c)
            del new_scores[c]

        new_profile = remove_and_condense_rank_profile(
            removed=eliminated_this_round,
            profile=profile,
            remove_zero_weight_ballots=False,
            remove_empty_ballots=False,
        )

        if store_states:
            remaining = score_dict_to_ranking(new_scores)
            new_state = ElectionState(
                round_number=prev_state.round_number + 1,
                eliminated=(frozenset(eliminated_this_round),),
                remaining=remaining,
                scores=new_scores,
            )

            self.election_states.append(new_state)
        return new_profile
