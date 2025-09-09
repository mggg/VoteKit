from votekit.elections.election_types.ranking.abstract_ranking import RankingElection
from votekit.pref_profile import RankProfile
from votekit.ballot import RankBallot
from votekit.elections.election_state import ElectionState
from votekit.utils import (
    first_place_votes,
    score_dict_to_ranking,
    tiebreak_set,
)
from votekit.cleaning import remove_and_condense_ranked_profile
import numpy as np
from typing import Optional, Literal
from functools import partial


class PluralityVeto(RankingElection):
    """
    Scores each candidate by their plurality (number of first) place votes,
    then in a randomized order it lets each voter decrement the score of their
    least favorite candidate. The candidate with the largest score at the end is
    then chosen as the winner.

    Args:
      profile (RankProfile): RankProfile to run election on. Note that
        ballots must have integer weights to be considered valid for this mechanism.
      m (int): Number of seats to elect.
      tiebreak (str, optional): Tiebreak method to use. Options are None, 'random', and 'borda'.
            Defaults to None, in which case a tie raises a ValueError.
      fpv_tie_convention (Literal["high", "average", "low"], optional): How to award points
            for tied first place votes. Defaults to "average", where if n candidates are tied for
            first, each receives 1/n points. "high" would award them each one point, and "low" 0.
            Only used by ``score_function`` parameter.

    Attributes:
        m (int): The number of seats to be filled in the election.
        tiebreak (Optional[str]): The tiebreak method selected for the election. Could be None,
            'random', or 'borda'.
        ballot_list (List[Ballot]): A list of Ballot instances representing the decondensed ballots
            where each ballot has a weight of 1.
        random_order (List[int]): A list of indices representing the randomized order in which
            voters' preferences are processed.
        preference_index (List[int]): A list of integers where each entry corresponds to the
            index of the least preferred candidate for each ballot.
        eliminated_dict (Dict[str, bool]): A dictionary mapping each candidate to a boolean value
            indicating whether the candidate has been eliminated.

    Raises:
        ValueError: If ``m`` is not positive or if it exceeds the number of candidates.
    """

    def __init__(
        self,
        profile: RankProfile,
        m: int,
        tiebreak: Optional[str] = None,
        fpv_tie_convention: Literal["high", "low", "average"] = "average",
    ):
        """
        Initializes the Plurality Veto election class.

        This constructor sets up the initial state of the election, including
        validating the profile, decondensing ballots (necessary for the running the steps of the
        election), and preparing other necessary data structures.

        Args:
            profile (RankProfile): The preference profile to be used in the election.
            m (int): The number of seats to be filled in the election.
            tiebreak (Optional[str]): The method used to resolve ties. Defaults to None.

        Raises:
            ValueError: If ``m`` is less than or equal to 0, or if ``m`` exceeds the number of
                candidates in the profile.
            AttributeError: If there are any ballots with ties and no tiebreak method is specified.
            TypeError: If the profile contains invalid ballots.
        """
        self._pv_validate_profile(profile)

        if m <= 0:
            raise ValueError("m must be positive.")
        elif len(profile.candidates_cast) < m:
            raise ValueError("Not enough candidates received votes to be elected.")

        self.m = m
        self.tiebreak = tiebreak

        if self.tiebreak is None and profile.ballots is not None:
            for ballot in profile.ballots:
                if ballot.ranking is not None and any(
                    len(s) > 1 for s in ballot.ranking
                ):
                    raise AttributeError(
                        "Found Ballots with ties but no tiebreak method was specified."
                    )

        # Ungroup ballots
        ballots = profile.ballots
        new_ballots = [RankBallot() for _ in range(int(profile.total_ballot_wt))]
        bidx = 0
        for b in ballots:
            for _ in range(int(b.weight)):
                new_ballots[bidx] = RankBallot(
                    ranking=b.ranking,
                    weight=1,
                    voter_set=b.voter_set,
                )
                bidx += 1

        profile = RankProfile(ballots=tuple(new_ballots), candidates=profile.candidates)

        self.ballot_list = profile.ballots
        self.random_order = list(range(int(profile.num_ballots)))
        np.random.shuffle(self.random_order)
        self.preference_index = [
            len(ballot.ranking) - 1 if ballot.ranking else -1
            for ballot in self.ballot_list
        ]

        self.eliminated_dict = {c: False for c in profile.candidates}

        super().__init__(
            profile,
            score_function=partial(
                first_place_votes, tie_convention=fpv_tie_convention
            ),
        )

    def _pv_validate_profile(self, profile: RankProfile):
        """
        Validate that each ballot has a ranking and that each
        ballot has integer weight.
        """
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
        Run one step of an election from the given profile and previous state.
        If m candidates have not yet been elected:
        if the current round is 0, count plurality scores and remove all
        candidates with a score of 0 to eliminated. Otherwise, this method
        eliminates a single candidate by decrementing scores of candidates on
        the current ticket until someone falls to a score of 0. Once we find
        that there are exactly m candidates remaining on the ticket, elect all
        of them.

        Args:
            profile (RankProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.
            store_states (bool, optional): True if `self.election_states` should be updated with the
                ElectionState generated by this round. This should only be True when used by
                `self._run_election()`. Defaults to False.

        Returns:
            RankProfile: The profile of ballots after the round is completed.
        """
        remaining_count = len(self.eliminated_dict) - sum(self.eliminated_dict.values())

        if remaining_count == self.m:
            # move all to elected, this is the last round
            elected = prev_state.remaining
            new_profile = RankProfile()

            if store_states:
                if self.score_function:
                    scores = self.score_function(new_profile)
                new_state = ElectionState(
                    round_number=prev_state.round_number + 1,
                    elected=elected,
                    scores=scores,
                )

            self.election_states.append(new_state)

        else:
            tiebreaks = {}
            new_scores = {c: s for c, s in prev_state.scores.items()}

            eliminated_cands = []
            if prev_state.round_number == 0:
                eliminated_cands = [
                    c for c, score in prev_state.scores.items() if score <= 0
                ]

            for rand_index, ballot_index in enumerate(self.random_order):
                if not self.preference_index[ballot_index] < 0:
                    ballot = self.ballot_list[ballot_index]
                    last_place = self.preference_index[ballot_index]

                    if ballot.ranking:
                        if len(ballot.ranking[last_place]) > 1:
                            if self.tiebreak:
                                tiebroken_ranking = tiebreak_set(
                                    ballot.ranking[last_place], profile, self.tiebreak
                                )
                            tiebreaks = {ballot.ranking[last_place]: tiebroken_ranking}
                        else:
                            tiebroken_ranking = (ballot.ranking[last_place],)

                    least_preferred = list(tiebroken_ranking[-1])[0]
                    new_scores[least_preferred] -= 1

                    if new_scores[least_preferred] <= 0:
                        eliminated_cands.append(least_preferred)
                        break

            # Circularly shift the randomized order array so that
            # we can continue where we left off in the next round
            self.random_order = (
                self.random_order[rand_index + 1 :]
                + self.random_order[: rand_index + 1]
            )

            for c in eliminated_cands:
                self.eliminated_dict[c] = True

            new_profile = remove_and_condense_ranked_profile(
                eliminated_cands,
                profile,
                remove_zero_weight_ballots=False,
                remove_empty_ballots=False,
            )

            self.ballot_list = new_profile.ballots

            self.preference_index = [
                len(ballot.ranking) - 1 if ballot.ranking else -1
                for ballot in self.ballot_list
            ]

            if store_states:
                eliminated = (frozenset(eliminated_cands),)

                score_ballots = tuple(
                    ballot for ballot in new_profile.ballots if ballot.ranking
                )
                score_profile = RankProfile(ballots=score_ballots)

                if self.score_function:
                    scores = self.score_function(score_profile)

                remaining = score_dict_to_ranking(scores)
                new_state = ElectionState(
                    round_number=prev_state.round_number + 1,
                    eliminated=eliminated,
                    remaining=remaining,
                    scores=scores,
                    tiebreaks=tiebreaks,
                )

                self.election_states.append(new_state)
        return new_profile
