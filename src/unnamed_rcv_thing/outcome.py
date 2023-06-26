class Outcome:
    """
    a class that represents the election outcome
    """
    def __init__(self, elected, eliminated, remaining, rankings):
        """
        Args:
            elected (a set of Candidate): candidates who pass a certain threshold to win an election
            eliminated (a set of Candidate): candidates who were eliminated (lost in the election)
            remaining (a set of Candidate): candidates who are still in the running
            rankings (a list of a set of Candidate): ranking of candidates with sets representing ties
        """
        self.elected = elected
        self.eliminated = eliminated
        self.remaining = remaining
        self.rankings = rankings
    