import pandas as pd
from pydantic import BaseModel
from typing import Optional

from .pref_profile import PreferenceProfile
from .utils import candidate_position_dict

pd.set_option("display.colheader_justify", "left")


class ElectionState(BaseModel):
    """
    Class for storing information on each round of an election and the final outcome.

    **Attributes**
    `curr_round`
    :   current round number. Defaults to 0.

    `elected`
    :   list of candidates who pass a threshold to win.

    `eliminated`
    :   list of candidates who were eliminated.

    `remaining`
    :   list of candidates who are still in the running.

    `profile`
    :   an instance of a PreferenceProfile object.

    `previous`
    :   an instance of ElectionState representing the previous round.

    **Methods**
    """

    curr_round: int = 0
    elected: list[set[str]] = []
    eliminated: list[set[str]] = []
    remaining: list[set[str]] = []
    profile: PreferenceProfile
    scores: dict = {}
    previous: Optional["ElectionState"] = None

    class Config:
        allow_mutation = False

    def get_all_winners(self) -> list[set[str]]:
        """
        Returns:
         A list of elected candidates ordered from first round to current round.
        """
        if self.previous:
            return self.previous.get_all_winners() + self.elected
        else:
            return self.elected

    def get_all_eliminated(self) -> list[set[str]]:
        """
        Returns:
          A list of eliminated candidates ordered from current round to first round.
        """
        if self.previous:
            return self.eliminated + self.previous.get_all_eliminated()
        else:
            return self.eliminated

    def get_rankings(self) -> list[set[str]]:
        """
        Returns:
          List of all candidates in order of their ranking after each round, first the winners,\
          then the eliminated candidates.
        """
        if self.remaining != [{}]:
            return self.get_all_winners() + self.remaining + self.get_all_eliminated()
        else:
            return self.get_all_winners() + self.get_all_eliminated()

    def get_round_outcome(self, round: int) -> dict:
        # {'elected':list[set[str]], 'eliminated':list[set[str]]}
        """
        Finds the outcome of a given round.

        Args:
            roundNum (int): Round number.

        Returns:
          A dictionary with elected and eliminated candidates.
        """
        if self.curr_round == round:
            return {
                "Elected": [c for s in self.elected for c in s],
                "Eliminated": [c for s in self.eliminated for c in s],
            }
        elif self.previous:
            return self.previous.get_round_outcome(round)
        else:
            raise ValueError("Round number out of range")

    def get_scores(self, round: int = curr_round) -> dict:
        """
        Returns a dictionary of the candidate scores for the inputted round.
        Defaults to the last round
        """
        if round == 0 or round > self.curr_round:
            raise ValueError('Round number out of range"')

        if round == self.curr_round:
            return self.scores

        return self.previous.get_scores(self.curr_round - 1)  # type: ignore

    def changed_rankings(self) -> dict:
        """
        Returns:
            A dictionary with keys = candidate(s) who changed \
                ranking from previous round and values = a tuple of (previous rank, new rank).
        """

        if not self.previous:
            raise ValueError("This is the first round, cannot compare previous ranking")

        prev_ranking: dict = candidate_position_dict(self.previous.get_rankings())
        curr_ranking: dict = candidate_position_dict(self.get_rankings())
        if curr_ranking == prev_ranking:
            return {}

        changes = {}
        for candidate, index in curr_ranking.items():
            if prev_ranking[candidate] != index:
                changes[candidate] = (prev_ranking[candidate], index)
        return changes

    def status(self) -> pd.DataFrame:
        """
        Returns:
          Data frame displaying candidate, status (elected, eliminated,
            remaining), and the round their status updated.
        """
        all_cands = [c for s in self.get_rankings() for c in s]
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
        return show.to_string(index=False, justify="justify")

    __repr__ = __str__
