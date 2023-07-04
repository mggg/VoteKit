from abc import ABC, abstractmethod


class Election(ABC):
    """
    Abstract class for election types, based on RCV base class
    implementation in rcv_cruncher
    """

    @abstractmethod
    def run_step(self):
        """
        Runs one 'step' or round of an election
        """
        pass

    # @abstractmethod
    # def compute_fp_votes(self) -> None:
    #     ''''
    #     Calculate the winners based on election style for a given round
    #     '''
    #     pass

    @abstractmethod
    def is_complete(self) -> bool:
        """
        Returns true or false if a round has completed an election
        """
        pass

    # def transfers(self) -> None:
    #     '''
    #     Transfers votes after a candidate has won or been eliminated
    #     '''
    #     pass

    # def update_preference_profile(self) -> None:
    #     ''''
    #     Makes necessary updates to preference profiles, removing ballots
    #     or candidates who have been elected or eliminated

    #     Maybe the same as transfers??
    #     '''
