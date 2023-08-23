import pandas as pd
from pydantic import BaseModel
from typing import Optional

from .pref_profile import PreferenceProfile


pd.set_option("display.colheader_justify", "left")


class ElectionState(BaseModel):
    """
    Object that stores information on each round of a RCV election and the final outcome.
    :param curr_round: :class:`int` : current round number. Defaults to 0 before an election.
    :param elected: :class:`list[str]` list of candidates who pass a threshold to win an election
    :param eliminated: :class:`list[str]` list of candidates who were eliminated
    :param remaining: :class:`list[str]` list of candidates who are still in the running
    :param rankings: :class: `list[set]` list ranking of candidates with sets representing ties
    :param profile: :class:`PreferenceProfile` an instance of a preference profile object
    :param previous: an instance of :class:`ElectionState` representing previous round
    """

    curr_round: int = 0
    elected: list = []
    eliminated: list = []
    remaining: list = []
    profile: PreferenceProfile
    previous: Optional["ElectionState"] = None

    class Config:
        allow_mutation = False

    def get_all_winners(self) -> list[str]:
        """
        Returns a list of elected candidates ordered from first round to current round.
        :rtype: :class:`list[str]`
        """
        if self.previous:
            return self.previous.get_all_winners() + self.elected
        else:
            return self.elected

    def get_all_eliminated(self) -> list[str]:
        """
        Returns a list of eliminated candidates ordered from current round to first round
        :rtype: :class:`list[str]`
        """
        elim = self.eliminated.copy()
        elim.reverse()
        if self.previous:
            elim += self.previous.get_all_eliminated()
        return elim

    def get_rankings(self) -> list[str]:
        """
        Returns list of all candidates in order of their ranking after each round
        :rtype: :class:`list[str]`
        """

        return self.get_all_winners() + self.remaining + self.get_all_eliminated()

    def get_round_outcome(self, roundNum: int) -> dict:
        # {'elected':list[str], 'eliminated':list[str]}
        """
        returns a dictionary with elected and eliminated candidates
        :rtype: :class:`dict`
        """
        if self.curr_round == roundNum:
            return {"Elected": self.elected, "Eliminated": self.eliminated}
        elif self.previous:
            return self.previous.get_round_outcome(roundNum)
        else:
            raise ValueError("Round number out of range")

    def changed_rankings(self) -> dict:
        """
        Returns dict of (key) string candidates who changed
        ranking from previous round and (value) a tuple of (previous rank, new rank)
        :rtype: :class:`dict`
        """

        if not self.previous:
            raise ValueError("This is the first round, cannot compare previous ranking")

        prev_ranking: dict = {
            cand: index for index, cand in enumerate(self.previous.get_rankings())
        }
        curr_ranking: dict = {
            cand: index for index, cand in enumerate(self.get_rankings())
        }
        if curr_ranking == prev_ranking:
            return {}

        changes = {}
        for candidate, index in curr_ranking.items():
            if prev_ranking[candidate] != index:
                changes[candidate] = (prev_ranking[candidate], index)
        return changes

    def status(self) -> pd.DataFrame:
        """
        Returns dataframe displaying candidate, status (elected, eliminated,
        remaining)
        :rtype: :class:`DataFrame`
        """
        all_cands = self.get_rankings()
        status_df = pd.DataFrame(
            {
                "Candidate": all_cands,
                "Status": ["Remaining"] * len(all_cands),
                "Round": [self.curr_round] * len(all_cands),
            }
        )

        for round in range(1, self.curr_round + 1):
            results = self.get_round_outcome(round)
            for status, candidates in results.items():
                for cand in candidates:
                    status_df.loc[status_df["Candidate"] == cand, "Status"] = status
                    status_df.loc[status_df["Candidate"] == cand, "Round"] = round

        return status_df

    def __str__(self):
        show = self.status()
        # show["Round"] = show["Round"].astype(str).str.rjust(3)
        # show["Status"] = show["Status"].str.ljust(10)
        return show.to_string(index=False, justify="justify")

    __repr__ = __str__

    ###############################################################################################

    # def add_winners_and_losers(self, winners: set[str], losers: set[str]) -> "Outcome":
    #   # example method, feel free to delete if not useful
    #  if not winners.issubset(self.remaining) or not losers.issubset(self.remaining):
    #     missing = (winners.difference(set(self.remaining)) |
    # (losersdifference(set(self.remaining)))
    #     raise ValueError(f"Cannot promote winners, {missing} not in remaining")
    # return Outcome(
    #     remaining=set(self.remaining).difference(winners | losers),
    #     elected=list(set(self.elected) | winners)
    #     eliminated=list(set(self.eliminated) | losers)
    # )

    #   def difference_remaining_candidates(
    #       self, prevOutcome1: "Outcome", prevOutcome2: "Outcome"
    #   ) -> float:
    #       """returns the fractional difference in number of
    #       remaining candidates; assumes ballots don't change by round
    #       """
    #       if (not prevOutcome1.get_profile()) or (not prevOutcome2.get_profile()):
    #           raise ValueError("Profile missing")
    # check if from same conshow
    #        elif set(prevOutcome1.get_profile().ballots) != set(
    #            prevOutcome2.get_profile().ballots
    #        ):
    #            raise ValueError("Cannot compare outcomes from different elections")
    #
    #        remaining_diff = len(
    #            (set(prevOutcome1.remaining)).difference(prevOutcome2.remaining)
    #        )
    #        allcandidates = len(prevOutcome1.get_profile().get_candidates())
    #        return remaining_diff / allcandidates
