from abc import ABC, abstractmethod
from .elections import ElectionState
from .pref_profile import PreferenceProfile
import pandas as pd
from .utils import (
    score_dict_to_ranking,
)
from typing import Callable, Optional, Union
from fractions import Fraction


class Election(ABC):
    """
    Abstract base class for election types.

    Args:
        profile (PreferenceProfile): The initial profile of ballots.
        score_function (Callable[[PreferenceProfile], Union[dict[str, Fraction], dict[str, float]]], optional):
            A function that converts profiles to a score dictionary mapping candidates to
            their current score. Used in creating ElectionState objects and sorting candidates in
            Round 0. If None, no score dictionary is saved and all candidates are tied in Round 0.
            Defaults to None.
        sort_high_low (bool, optional): How to sort candidates based on `score_function`. True sorts
            from high to low. Defaults to True.

    Attributes:
        election_states (list[ElectionState]): A list of election states, one for each round of
            the election. The list is 0 indexed, so the initial state is stored at index 0, round 1
            at 1, etc.
        score_function (Callable[[PreferenceProfile], Union[dict[str, Fraction], dict[str, float]]], optional):
            A function that converts profiles to a score dictionary mapping candidates to
            their current score. Used in creating ElectionState objects. Defaults to None.
        length (int): The number of rounds of the election.
    """  # noqa

    def __init__(
        self,
        profile: PreferenceProfile,
        score_function: Optional[
            Callable[[PreferenceProfile], Union[dict[str, Fraction], dict[str, float]]]
        ] = None,
        sort_high_low: bool = True,
    ):
        self._validate_profile(profile)
        self._profile = profile
        self.score_function = score_function
        self.sort_high_low = sort_high_low
        self.election_states: list[ElectionState] = []
        self._run_election()
        self.length = len(self.election_states) - 1

    def get_profile(self, round_number: int = -1) -> PreferenceProfile:
        """
        Fetch the PreferenceProfile of the given round number.

        Args:
            round_number (int, optional): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            PreferenceProfile

        """
        if (
            round_number < -len(self.election_states)
            or round_number > len(self.election_states) - 1
        ):
            raise IndexError("round_number out of range.")

        round_number = round_number % len(self.election_states)

        profile = self._profile

        for i in range(round_number):
            profile = self._run_step(profile, self.election_states[i])

        return profile

    def get_step(
        self, round_number: int = -1
    ) -> tuple[PreferenceProfile, ElectionState]:
        """
        Fetches the profile and ElectionState of the given round number.

        Args:
            round_number (int, optional): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            tuple[PreferenceProfile, ElectionState]
        """
        return (self.get_profile(round_number), self.election_states[round_number])

    def get_elected(self, round_number: int = -1) -> tuple[frozenset[str], ...]:
        """
        Fetch the elected candidates up to the given round number.

        Args:
            round_number (int, optional): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            tuple[frozenset[str],...]:
                List of winning candidates in order of election. Candidates
                in the same set were elected simultaneously, i.e. in the final ranking
                they are tied.
        """
        if (
            round_number < -len(self.election_states)
            or round_number > len(self.election_states) - 1
        ):
            raise IndexError("round_number out of range.")

        round_number = round_number % len(self.election_states)

        return tuple(
            [
                s
                for state in self.election_states[: (round_number + 1)]
                for s in state.elected
                if state.elected != (frozenset(),)
            ]
        )

    def get_eliminated(self, round_number: int = -1) -> tuple[frozenset[str], ...]:
        """
        Fetch the eliminated candidates up to the given round number.

        Args:
            round_number (int, optional): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            tuple[frozenset[str],...]:
                Tuple of eliminated candidates in reverse order of elimination.
                Candidates in the same set were eliminated simultaneously, i.e. in the final ranking
                they are tied.
        """
        if (
            round_number < -len(self.election_states)
            or round_number > len(self.election_states) - 1
        ):
            raise IndexError("round_number out of range.")

        round_number = round_number % len(self.election_states)

        # reverses order to match ranking convention
        return tuple(
            [
                s
                for state in self.election_states[round_number::-1]
                for s in state.eliminated[::-1]
                if state.eliminated != (frozenset(),)
            ]
        )

    def get_remaining(self, round_number: int = -1) -> tuple[frozenset[str], ...]:
        """
        Fetch the remaining candidates after the given round.

        Args:
            round_number (int, optional): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            tuple[frozenset[str],...]:
                Tuple of sets of remaining candidates. Ordering of tuple
                denotes ranking of remaining candidates, sets denote ties.
        """
        return tuple(self.election_states[round_number].remaining)

    def get_ranking(self, round_number: int = -1) -> tuple[frozenset[str], ...]:
        """
        Fetch the ranking of candidates after a given round.

        Args:
            round_number (int, optional): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            tuple[frozenset[str],...]: Ranking of candidates.
        """
        # len condition handles empty remaining candidates
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
        Yield the status (elected, eliminated, remaining) of the candidates in the given round.
        DataFrame is sorted by current ranking.

        Args:
            round_number (int, optional): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            pd.DataFrame:
                Data frame displaying candidate, status (elected, eliminated,
                remaining), and the round their status updated.
        """
        if (
            round_number < -len(self.election_states)
            or round_number > len(self.election_states) - 1
        ):
            raise IndexError("round_number out of range.")

        round_number = round_number % len(self.election_states)

        new_index = [c for s in self.get_ranking(round_number) for c in s]

        if round_number == -1:
            round_number = len(self.election_states) - 1

        candidates = self._profile.candidates

        status_df = pd.DataFrame(
            {"Status": ["Remaining"] * len(candidates), "Round": [0] * len(candidates)},
            index=candidates,
        )

        for i in range(round_number):
            state = self.election_states[i + 1]

            for s in state.elected:
                for c in s:
                    status_df.at[c, "Status"] = "Elected"
                    status_df.at[c, "Round"] = i + 1

            for s in state.eliminated:
                for c in s:
                    status_df.at[c, "Status"] = "Eliminated"
                    status_df.at[c, "Round"] = i + 1

            for s in state.remaining:
                for c in s:
                    status_df.at[c, "Round"] = i + 1

        status_df = status_df.reindex(new_index)
        return status_df

    @abstractmethod
    def _validate_profile(self, profile: PreferenceProfile):
        """
        Validate that a profile contains appropriate ballots for election. Raises
        TypeError if not.

        Args:
            profile (PreferenceProfile): Profile of ballots.
        """
        pass

    @abstractmethod
    def _run_step(
        self, profile: PreferenceProfile, prev_state: ElectionState, store_states=False
    ) -> PreferenceProfile:
        """
        Run one step of an election from the given profile and previous state.

        Args:
            profile (PreferenceProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.
            store_states (bool, optional): True if `self.election_states` should be updated with the
                ElectionState generated by this round. This should only be True when used by
                `self._run_election()`. Defaults to False.

        Returns:
            PreferenceProfile: The profile of ballots after the round is completed.
        """
        pass

    @abstractmethod
    def _is_finished(self) -> bool:
        """
        Returns True if election is finished, False if another round is needed.
        """
        pass

    def _run_election(self):
        """
        Runs the election.
        """
        profile = self._profile

        scores = {}
        remaining = profile.candidates

        if self.score_function:
            # compute scores and sort
            scores = self.score_function(profile)
            remaining = score_dict_to_ranking(scores, self.sort_high_low)

        else:
            # if no scores, all candidates are tied
            remaining = (frozenset(remaining),)

        self.election_states.append(ElectionState(remaining=remaining, scores=scores))

        while not self._is_finished():
            profile = self._run_step(
                profile, self.election_states[-1], store_states=True
            )

    def __len__(self):
        return self.length

    def __str__(self):
        return self.get_status_df().to_string(index=True, justify="justify")

    __repr__ = __str__
