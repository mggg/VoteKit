from abc import ABC, abstractmethod
from typing import Any

from .elections import ElectionState
from .pref_profile import PreferenceProfile
from .utils import recursively_fix_ties, fix_ties


class Election(ABC):
    """
    Abstract base class for election types. Includes functions to resolve input
    ties included in PreferenceProfile.

    Attributes:
        profile: a PreferenceProfile.
        ballot_ties: an optional Bool, defaults to True. If True, resolve ties on ballots.
    """

    def __init__(self, profile: PreferenceProfile, ballot_ties: bool = True):
        if ballot_ties:
            self._profile = self.resolve_input_ties(profile)
        else:
            self._profile = profile

        self.state = ElectionState(curr_round=0, profile=self._profile)

    def reset(self):
        """
        Reset the ElectionState object to initial conditions.
        """

        self.state = ElectionState(curr_round=0, profile=self._profile)

    def run_to_step(self, step: int):
        """
        Run the election to the given step.

        Args:
            step (int): The step of the election to run to.

        Returns:
            The ElectionState object for round = step.
        """
        if isinstance(step, int):
            if step < 0:
                raise ValueError("Step must be a non-negative integer.")

            elif step >= 0:
                while self.state.curr_round < step:
                    self.run_step()

                while self.state.curr_round > step:
                    if self.state.previous:
                        self.state = self.state.previous
                    else:
                        raise ValueError("Previous state is None type.")

            return self.state

    @abstractmethod
    def run_step(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def run_election(self, *args: Any, **kwargs: Any):
        pass

    def resolve_input_ties(self, profile: PreferenceProfile) -> PreferenceProfile:
        """
        Takes in a PeferenceProfile with potential ties in a ballot. Replaces
        ballots with ties with fractionally weighted ballots corresonding to
        all permutation of the tied ranking

        Args:
            profile: Input profile with potentially tied rankings

        Returns:
            A PreferenceProfile with resolved ties
        """
        new_ballots = []
        old_ballots = profile.get_ballots()

        for ballot in old_ballots:
            if not any(len(rank) > 1 for rank in ballot.ranking):
                new_ballots.append(ballot)
            else:
                num_ties = 0
                for rank in ballot.ranking:
                    if len(rank) > 1:
                        num_ties += 1

                resolved_ties = fix_ties(ballot)
                new_ballots += recursively_fix_ties(resolved_ties, num_ties)

        return PreferenceProfile(ballots=new_ballots)
