from abc import ABC, abstractmethod
from typing import Union, Any
from .profile import PreferenceProfile
from .election_state import ElectionState
from .ballot_generator import BallotGenerator

# Hyperparams:

# W/C share (1 num) == slate_voter_prop
# Cohesion: w for w and poc for poc (2 numbers) = cohesion
# Concentration/alpa: DL params (4 numbers) == alpha

# election functions
# Number of seats
# Number of w/c candidates


class Simulation(ABC):
    """
    Base class for model complex elections or statewide simulations as
    done in MGGG's RCV research.
    """

    def __init__(self, ballots: Union[PreferenceProfile, dict]):
        if ballots:
            self.ballots = ballots

    @abstractmethod
    def run_simulation(self) -> Any:
        """
        User written function to feed parameters into election simulation
        """
        pass

    @abstractmethod
    def sim_election(self) -> ElectionState:
        """
        Runs election(s) with specified parameters.
        """
        pass

    def generate_ballots(
        self, num_ballots: int, candidates: Union[list, dict], hyperparams: dict
    ) -> list[tuple[Any, PreferenceProfile]]:
        """
        Function that generates perference profiles if ballot generator model
        is assigned to the class.

        Can be overridden based on user needs.
        """
        if isinstance(self.ballots, PreferenceProfile):
            raise TypeError("No generator assigned to produce ballots")

        ballots = []
        for model_name, model in self.ballots:
            generator: BallotGenerator = model(num_ballots, candidates, hyperparams)
            ballots.append((model_name, generator.generate_profile()))

        return ballots
