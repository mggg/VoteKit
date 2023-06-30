import uuid

class Candidate:
    """
    a class that represents a candidate
    """
    def __init__(self, name, group=None):
        """
        Args:
            name (string): name of candidate
            group (string, optional): a category that the candidate belongs to 
            (ex. ‘Fringe’). Defaults to None.
        """
        self.id = uuid.uuid4()
        self.name = name
        self.group = group
    
    def __str__(self):
        return f"{self.name}"
     

class XCandidate(Candidate):
    """
    a class that represents a null ranking
    """
    def __init__(self):
        super().__init__(name='x', group='x')