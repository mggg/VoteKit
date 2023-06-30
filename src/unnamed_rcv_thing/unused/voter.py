import uuid

class Voter:
    """
    A class that represents a voter
    """
    def __init__(self, candidate_ranking, name='', candidate_scores=None):
        """
        Args:
            name (string, option): name of the voter. Defaults to ''
            candidate_ranking (list of Candidate): the candidates ordered by the voter's ranking
            candidate_scores (list of double, optional): 
            the weights the voter assigns to each candidate's ranking. Defaults to None.
        """
        self.id = uuid.uuid4()
        self.name = name
        self.candidate_ranking = candidate_ranking
        self.candidate_scores = candidate_scores
    
    def __str__(self):
        return f"{self.name} : {self.candidate_ranking}"