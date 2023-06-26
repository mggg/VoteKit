import uuid

class PreferenceProfile:
    """
    a class that represents a preference profile / schedule for an election
    """
    def __init__(self, ballots, candidates):
        """
        Args:
            ballots (list of Ballot): a list of ballots in the election
            candidates (list of Candidates): a list of candidates in the election
        """
        self.id = uuid.uuid4()
        self.ballots = ballots
        self.candidates = candidates
        self.ballot_weights = [b.score for b in ballots]