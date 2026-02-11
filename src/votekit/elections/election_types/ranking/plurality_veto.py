import random
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

TILDE = frozenset("~")


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

    Attributes:
        m (int): The number of seats to be filled in the election.
        tiebreak (str): The tiebreak method selected for the election.
            One of 'first_place', 'random', 'borda', or 'lex'.
        scoring_tie_convention (str): Method for awarding points for tied first-place votes.
        candidates (frozenset[str]): The set of candidates in the election.
        eliminated (set[str]): The set of candidates that have been eliminated.
        num_voters (int): The total number of voters in the election.
            Calculated as the sum of the weights of all ballots.
        tiebreak_order (Optional[tuple[frozenset[str]]]): The candidate ordering used to break last-place ties
            when processing vetoes. ``None`` if ``tiebreak`` = 'random'.

    Raises:
        ValueError: If ``m`` is less than or equal to 0, or if ``m`` exceeds the number of
            candidates in the profile.
        AttributeError: If there are any ballots with ties and no tiebreak method is specified.
        TypeError: If the profile contains invalid ballots.
    """

    def __init__(
        self,
        profile: RankProfile,
        m: int = 1,
        tiebreak: Literal["first_place", "borda", "random", "lex"] = "first_place",
        scoring_tie_convention: Literal["high", "low", "average"] = "average",
    ):
        # validate input
        self.m = m
        self.tiebreak: Literal["first_place", "borda", "random", "lex"] = tiebreak
        self.scoring_tie_convention: Literal["high", "low", "average"] = (
            scoring_tie_convention
        )
        self._pv_validate_input(profile)

        # group profile in case of repeated ballots
        profile = profile.group_ballots()
        # this also gives us our own copy of profile.df,
        # so we can modify that in place
        self._df = profile.df
        self.candidates = frozenset(profile.candidates)
        self.eliminated = set()

        self._max_ranking_length = profile.max_ranking_length

        self.num_voters = int(profile.total_ballot_wt)
        self._voter_order = iter(np.random.permutation(self.num_voters))

        # a ballot with weight w is a bin containing w voters
        # using cumulative sum defines the boundaries of each bin, so we can index against this column
        self._df["cumsum"] = self._df["Weight"].cumsum()

        # calculate the candidates not mentioned for each ballot
        # we consider them to be tied for last place
        self._unlisted_candidates = tuple(
            self.candidates - (frozenset.union(*ballot[: self._max_ranking_length]))
            for ballot in self._df.itertuples(index=False)
        )

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
            # and convert it to a list, which we keep hidden because, for efficiency, we will
            # remove candidates as they are eliminated
            self._tiebreak_order = [next(iter(s)) for s in self.tiebreak_order]

        super().__init__(
            profile,
            score_function=partial(
                first_place_votes, tie_convention=scoring_tie_convention
            ),
        )

    def _get_ballot_idx(self, voter_idx: int) -> np.intp:
        """
        Converts a voter index in [0, num_voters) to a ballot index in [0, len(self._df)) that can
        be used to retrieve the row in self._df corresponding to that voter's ballot.
        """
        ballot_idx = np.searchsorted(self._df["cumsum"], voter_idx, side="right")
        if ballot_idx >= len(self._df):
            raise IndexError(
                f"Voter index {voter_idx} out of range for a profile of {len(self._df)} voters."
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
        if len(candidate_set) == 1:
            return next(iter(candidate_set))
        if self.tiebreak == "random":
            return random.choice(list(candidate_set))
        assert self._tiebreak_order
        for c in reversed(self._tiebreak_order):
            if c in candidate_set:
                return c
        raise ValueError(
            f"Tried to tiebreak {candidate_set}, but no remaining candidate appeared in that set."
            f"Remaining candidates: {self._tiebreak_order}"
        )

    def _get_veto(self, ballot_idx: np.intp) -> str:
        """
        Given a ballot index, returns the candidate to veto.
        For deterministic tiebreak methods, returns the most recent veto
        that ballot gave; if that candidate has been eliminated, calculates
        the new veto and updates _veto_cache.
        When calculating veto, first considers unlisted candidates.
        If they have all been eliminated, then it walks backward through the ballot
        until it finds a remaining candidate. Breaks ties with _break_tie.

        Args:
            ballot_idx (int): A ballot index in [0, len(self._df)) corresponds to the row in
            self._df containing that ballot.

        Returns:
            veto (str): The candidate to be vetoed.

        Raises:
            IndexError: If ballot_idx is out of range.
        """
        # for deterministic tiebreaks, try to get the veto from the veto cache
        if self.tiebreak != "random":
            most_recent_veto = self._veto_cache[ballot_idx]
            if most_recent_veto and most_recent_veto not in self.eliminated:
                return most_recent_veto

        potential_vetoes = self._unlisted_candidates[ballot_idx] - self.eliminated
        if not potential_vetoes:
            # all unlisted candidates have been eliminated,
            # walk backward through ballot until we find a remaining candidate
            ballot = self._df.iloc[ballot_idx]
            ranking = ballot[: self._max_ranking_length]
            for potential_vetoes in reversed(ranking.values):
                potential_vetoes -= self.eliminated | TILDE
                if potential_vetoes:
                    break
            else:
                raise ValueError(
                    f"Ballot {ranking} was depleted before identifying a remaining candidate."
                )
        veto = self._break_tie(potential_vetoes)

        if self.tiebreak != "random":
            self._veto_cache[ballot_idx] = veto
        return veto

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

    def _is_finished(self) -> bool:
        cands_elected = [len(s) for s in self.get_elected()]
        return sum(cands_elected) >= self.m

    def _run_step(
        self, profile: RankProfile, prev_state: ElectionState, store_states=False
    ) -> RankProfile:
        """
        Runs one round of the PluralityVeto election.
        If there are only m candidates remaining, they are elected.
        If the current round is 0, each candidate's score equals the number of first-place votes
        they received. Start by eliminating any candidates who received no first-place votes.
        If we removed any candidates, move to the next round.
        Otherwise, and in all subsequent rounds, repeatedly query the next voter
        and identify their least favorite candidate, breaking ties as necessary,
        and decrement that candidate's score by 1 (a "veto").
        When any candidate's score reaches 0, eliminate them and proceed to the next round.

        Args:
            profile (RankProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.
            store_states (bool, optional): True if `self.election_states` should be updated with the
                ElectionState generated by this round. This should only be True when used by
                `self._run_election()`. Defaults to False.

        Returns:
            RankProfile: The profile of ballots after the round is completed.
        """
        remaining_count = len(self.candidates - self.eliminated)

        if remaining_count == self.m:
            # move all to elected, this is the last round
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

        new_scores = prev_state.scores.copy()

        eliminated_this_round = []
        # eliminate candidates that received no first-place votes
        if prev_state.round_number == 0:
            for c, score in new_scores.items():
                if score <= 0:
                    eliminated_this_round.append(c)

        # if we removed any candidates with 0 first-place votes, we proceed to the next round
        if not eliminated_this_round:
            # iterate over voters, processing vetoes, until some candidate is eliminated
            while True:
                voter_idx = next(self._voter_order)
                ballot_idx = self._get_ballot_idx(voter_idx)
                veto = self._get_veto(ballot_idx)

                # deduct one point for each veto
                new_scores[veto] -= 1
                # eliminate a candidate whose score reaches zero
                if new_scores[veto] <= 0:
                    eliminated_this_round.append(veto)
                    break

        for c in eliminated_this_round:
            self.eliminated.add(c)
            del new_scores[c]
            if self.tiebreak != "random":
                self._tiebreak_order.remove(c)

        new_profile = remove_and_condense_rank_profile(
            removed=eliminated_this_round,
            profile=profile,
            remove_zero_weight_ballots=False,
            remove_empty_ballots=False,
        )

        if store_states:
            eliminated = (frozenset(eliminated_this_round),)

            remaining = score_dict_to_ranking(new_scores)
            new_state = ElectionState(
                round_number=prev_state.round_number + 1,
                eliminated=eliminated,
                remaining=remaining,
                scores=new_scores,
            )

            self.election_states.append(new_state)
        return new_profile
