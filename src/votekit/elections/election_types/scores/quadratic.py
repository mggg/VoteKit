from .rating import GeneralRating
from votekit.pref_profile import ScoreProfile
import math
from typing import Optional
import copy

class Quadratic(GeneralRating):
    """
    Quadratic election where the cost of casting multiple votes for the same candidate increases quadratically.

    Each voter is given a fixed number of voting credits, which can be spent to support candidates. The 
    cost of casting multiple votes for the same candidate increases quadratically (i.e. n votes 
    cost n^2 credits). Note that the ballot score values can represent either votes or credits spend on 
    candidates, and you specify which it is in the boolean isCredits. Score values and k (the 
    credit budget) must be whole numbers. 

    Args:
        profile (ScoreProfile): Profile to conduct election on
        m (int, optional): Number of seats to elect. Defaults to 1.
        k (float): Total budget per voter. k must be a whole number. 
            In the case of Quadratic voting, this refers to the total budget of credits a voter can spend. 
        isCredits (boolean): isCredits = True means that scores represent credits and 
            isCredits = False means scores represent votes.
        tiebreak (str,optional): Tiebreak method to use. Options are None and 'random'.
            Defaults to None, in which case a tie raises a ValueError.
    """

    def __init__(
        self,
        profile: ScoreProfile,
        m: int = 1,
        k: float = None, 
        isCredits = False,
        tiebreak: Optional[str] = None,
    ):
        if not k.is_integer():
            raise TypeError(f"Credit budget k must be a whole number.")
        profile = self._check_credits(profile, k, isCredits)
        super().__init__(profile, m=m, k=k, tiebreak=tiebreak)

    def _check_credits(self, profile: ScoreProfile, k:float = None, isCredits=False):
        """
        Ensures that every ballot is within credit budget (k).
        Ensures that if ballots give credits, that credits are perfect squares 
        converts those credits into votes. 
        """
        profile_copy = copy.deepcopy(profile)
        for b in profile_copy.ballots:
            #check to make sure that scores are whole numbers
            if not all(isinstance(v, float) and v.is_integer() for v in b.scores.values()):
                raise TypeError(f"Scores must be whole numbers.")
            if isCredits: #isCredits -> scores mean credits
                #check to make sure credits are in budget
                if sum(b.scores.values()) > k:
                    raise TypeError(f"Ballot {b} violates credit budget {k}.")
                #check to make sure credits are perfect squares
                if not all(math.sqrt(v).is_integer() for v in b.scores.values()):
                    raise TypeError(f"Ballot {b} violates credit's perfect squares requirement.")
                #convert scores to votes (takes the square root of score values)
                for v in b.scores:
                    b.scores[v] = math.sqrt(b.scores[v])
            else: #not isCredits -> scores mean votes
                #convert scores (votes) to credits
                #check to make sure credits are in budget
                if sum(v**2 for v in b.scores.values()) > k:
                    raise TypeError(f"Ballot {b} violates credit budget {k}.")
        return profile_copy



