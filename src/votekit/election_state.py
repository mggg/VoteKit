import pandas as pd
from pydantic import BaseModel
from typing import Optional
import json

from .pref_profile import PreferenceProfile
from .utils import candidate_position_dict

pd.set_option("display.colheader_justify", "left")


class ElectionState(BaseModel):
    """
    Class for storing information on each round of an election and the final outcome.

    Args:
        profile (PreferenceProfile): An instance of a ``PreferenceProfile`` object.
        curr_round (int, optional): Current round number. Defaults to 0.
        elected (list[set[str]], optional): List of sets of candidates who pass a threshold to win.
            Candidates in the same set were elected simultaneously. Defaults to empty list.
        eliminated_cands (list[set[str]], optional): List of sets of candidates who were eliminated.
            Candidates in the same set were eliminated simultaneously. Defaults to empty list.
        remaining (list[set[str]], optional): List of sets of candidates who are remaining.
            Candidates in the same set were eliminated simultaneously. Defaults to empty list.
        scores (dict, optional): Dictionary mapping candidate names to scores. Defaults to
            empty dictionary.
        previous (ElectionState, optional): An instance of an ``ElectionState`` representing the
            previous round. Defaults to None.

    Attributes:
        profile (PreferenceProfile): An instance of a ``PreferenceProfile`` object.
        curr_round (int, optional): Current round number. Defaults to 0.
        elected (list[set[str]], optional): List of sets of candidates who pass a threshold to win.
            Candidates in the same set were elected simultaneously. Defaults to empty list.
        eliminated_cands (list[set[str]], optional): List of sets of candidates who were eliminated.
            Candidates in the same set were eliminated simultaneously. Defaults to empty list.
        remaining (list[set[str]], optional): List of sets of candidates who are remaining.
            Candidates in the same set were eliminated simultaneously. Defaults to empty list.
        scores (dict, optional): Dictionary mapping candidate names to scores. Defaults to
            empty dictionary.
        previous (ElectionState, optional): An instance of an ``ElectionState`` representing the
            previous round. Defaults to None.
    """

    curr_round: int = 0
    elected: list[set[str]] = []
    eliminated_cands: list[set[str]] = []
    remaining: list[set[str]] = []
    profile: PreferenceProfile
    scores: dict = {}
    previous: Optional["ElectionState"] = None

    class Config:
        allow_mutation = False

    def winners(self) -> list[set[str]]:
        """
        Winners up to current round.

        Returns:
            list[set[str]]: A list of elected candidates ordered from first round to current round.
        """
        if self.previous:
            return self.previous.winners() + self.elected

        return self.elected

    def eliminated(self) -> list[set[str]]:
        """
        Eliminated candidates up to current round.

        Returns:
            list[set[str]]: A list of eliminated candidates ordered from current round to first
                round.
        """
        if self.previous:
            return self.eliminated_cands + self.previous.eliminated()

        return self.eliminated_cands

    def rankings(self) -> list[set[str]]:
        """
        Show the current rankings of all candidates.

        Returns:
            list[set[str]]: List of all candidates in order of their ranking after each round,
                first the winners, then remaining, then the eliminated candidates.
        """
        if self.remaining != [{}]:
            return self.winners() + self.remaining + self.eliminated()

        return self.winners() + self.eliminated()

    def round_outcome(self, round: int) -> dict:
        """
        Finds the outcome of a given round.

        Args:
            round (int): Round number.

        Returns:
            dict: A dictionary with elected, remaining, and eliminated candidates.
        """
        if self.curr_round == round:
            return {
                "Elected": self.elected,
                "Eliminated": self.eliminated_cands,
                "Remaining": self.remaining,
            }
        elif self.previous:
            return self.previous.round_outcome(round)
        else:
            raise ValueError("Round number out of range")

    def get_scores(self, round: int = curr_round) -> dict:
        """
        Get the scores for a desired round.

        Args:
            round (int, optional): Round number. Defaults to current round.
        Returns:
            dict: A dictionary of the candidate scores for the inputted round.
        """
        if round == 0 or round > self.curr_round:
            raise ValueError('Round number out of range"')

        if round == self.curr_round:
            return self.scores

        return self.previous.get_scores(self.curr_round - 1)  # type: ignore

    def changed_rankings(self) -> dict:
        """
        Which candidates changed rank between current and previous round.

        Returns:
            dict: A dictionary with keys = candidate(s) who changed ranking from previous round
                and values = a tuple of (previous rank, new rank).
        """

        if not self.previous:
            raise ValueError("This is the first round, cannot compare previous ranking")

        prev_ranking: dict = candidate_position_dict(self.previous.rankings())
        curr_ranking: dict = candidate_position_dict(self.rankings())
        if curr_ranking == prev_ranking:
            return {}

        changes = {}
        for candidate, index in curr_ranking.items():
            if prev_ranking[candidate] != index:
                changes[candidate] = (prev_ranking[candidate], index)
        return changes

    def status(self) -> pd.DataFrame:
        """
        Yield the status of the candidates in the current round.

        Returns:
            pd.DataFrame: Data frame displaying candidate, status (elected, eliminated,
                remaining), and the round their status updated.
        """
        all_cands = [c for s in self.rankings() for c in s]
        status_df = pd.DataFrame(
            {
                "Candidate": all_cands,
                "Status": ["Remaining"] * len(all_cands),
                "Round": [self.curr_round] * len(all_cands),
            }
        )

        for round in range(1, self.curr_round + 1):
            results = self.round_outcome(round)
            for status, ranking in results.items():
                for s in ranking:
                    for cand in s:
                        tied_str = ""
                        # if tie
                        if len(s) > 1:
                            remaining_cands = ", ".join(list(s.difference(cand)))
                            tied_str = f" (tie with {remaining_cands})"

                        status_df.loc[status_df["Candidate"] == cand, "Status"] = (
                            status + tied_str
                        )
                        status_df.loc[status_df["Candidate"] == cand, "Round"] = round

        return status_df

    def to_dict(self, keep: list = []) -> dict:
        """
        Returns election results as a dictionary.

        Args:
            keep (list, optional): List of information to store in dictionary. Should be subset of
                "elected", "eliminated", "remaining", "ranking". Defaults to empty list,
                which stores all information.

        Returns:
            dict: Dictionary of election results.

        """
        keys = ["elected", "eliminated", "remaining", "ranking"]
        values: list = [
            self.winners(),
            self.eliminated(),
            self.remaining,
            self.rankings(),
        ]

        rv = {}
        for key, values in zip(keys, values):
            if keep and key not in keep:
                continue
            # pull out candidates from sets, if tied adds tuple of tied candidates
            temp_lst = []
            for cand_set in values:
                if len(cand_set) > 1:
                    temp_lst.append(tuple(cand_set))
                else:
                    temp_lst += [cand for cand in cand_set]
            rv[key] = temp_lst

        return rv

    def to_json(self, file_path: str, keep: list = []):
        """
        Saves election state object as a JSON file.

        Args:
            file_path (str): The name of the file path.
            keep (list, optional): List of information to store in dictionary, should be subset of
                "elected", "eliminated", "remaining", "ranking". Defaults to empty list,
                which stores all information.
        """

        json_dict = json.dumps(self.to_dict(keep=keep))
        with open(file_path, "w") as outfile:
            outfile.write(json_dict)

    def __str__(self):
        show = self.status()
        print(f"Current Round: {self.curr_round}")
        return show.to_string(index=False, justify="justify")

    __repr__ = __str__
