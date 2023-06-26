import uuid

class Ballot:  
    """
    a class that represents a ballot (a common type of voter based on candidate rankings)
    """
    def __init__(self, voters, candidate_ranking, score=None):
        """
        Args:
            voters (list of Voter): _description_
            candidate_ranking (list of Candidate): _description_
            score (int, optional): assigned weight to the ballot. Defaults to the number of voters with the same candidate ranking.
        """
        self.id = uuid.uuid4()
        self.voters = voters
        self.candidate_ranking = candidate_ranking
        self.is_spoiled = False

        if score:
            self.score = score
        else:
            self.score = len(voters)


    # TODO: define equality for Ballot and iterable
    
    