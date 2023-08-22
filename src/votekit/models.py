from abc import ABC, abstractmethod
from typing import Union, Any

from .ballot_generator import BallotGenerator
from .election_state import ElectionState
from .pref_profile import PreferenceProfile


class Simulation(ABC):
    """
    Base class for model complex elections or statewide simulations as
    done in MGGG's RCV research.

    *Attributes*

        Ballots: PreferenceProfile or dictionary of ballot generators
    """

    def __init__(self, ballots: Union[PreferenceProfile, dict, None] = None):
        if ballots:
            self.ballots = ballots

    @abstractmethod
    def run_simulation(self) -> Any:
        """
        User written function to feed parameters into election simulation
        """
        pass

    @abstractmethod
    def sim_election(self) -> Union[ElectionState, list]:
        """
        Runs election(s) with specified parameters.
        """
        pass

    def generate_ballots(
        self, num_ballots: int, candidates: Union[list, dict], params: dict
    ) -> list[tuple[Any, PreferenceProfile]]:
        """
        Generates perference profiles if ballot generator models
        are assigned to the class.

        Can be overridden based on user needs.
        """
        if isinstance(self.ballots, PreferenceProfile):
            raise TypeError("No generator assigned to produce ballots")

        ballots = []
        for model_name, model in self.ballots.items():
            generator: BallotGenerator = model(
                candidates=candidates,
                hyperparams=params,
            )
            ballots.append((model_name, generator.generate_profile(num_ballots)))

        return ballots
