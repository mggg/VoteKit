import copy
import math

from votekit.elections.election_types.scores.rating import GeneralRating
from votekit.pref_profile import ScoreProfile


class Quadratic(GeneralRating):
    """
    Quadratic election where the cost of casting multiple votes for the same candidate increases quadratically.

    Each voter is given a fixed number of voting credits, which can be spent to support candidates. The
    cost of casting multiple votes for the same candidate increases quadratically (i.e. n votes
    cost n^2 credits). Note that the ballot score values can represent either votes or credits spend on
    candidates, and you specify which it is in the boolean is_credits. Score values and k (the
    credit budget) must be whole numbers.

    Raises:
        ValueError: k must be a whole number

    Args:
        profile (ScoreProfile): Profile to conduct election on
        m (int, optional): Number of seats to elect. Defaults to 1.
        k (float): Total budget per voter. k must be a whole number.
            In the case of Quadratic voting, this refers to the total budget of credits a voter can spend.
        is_credits (boolean): is_credits = True means that scores represent credits and
            is_credits = False means scores represent votes.
        tiebreak (str,optional): Tiebreak method to use. Options are None and 'random'.
            Defaults to None, in which case a tie raises a ValueError.
    """

    def __init__(
        self,
        profile: ScoreProfile,
        m: int = 1,
        k: float = 1,
        is_credits=False,
        tiebreak: str | None = None,
    ):
        if isinstance(k, float) and not k.is_integer():
            raise ValueError(f"Credit budget k must be a whole number.")
        profile = self._check_credits(profile, k, is_credits)
        super().__init__(profile, m=m, k=k, L=int(math.sqrt(k)), tiebreak=tiebreak)
        # super().__init__(profile, m=m, k=k, L=k, tiebreak=tiebreak)

    def _check_credits(self, profile: ScoreProfile, k=1, is_credits=False):
        """
        Ensures that every ballot is within credit budget (k).

        No matter if ballot scores was given in credits or votes, ensures that all ballots remain within
        the budget.

        Args:
            profile (PreferenceProfile): Profile to validate.
            k (float): Total budget per voter
            is_credits (boolean): dictates whether scores are credits or votes

        Raises:
            ValueError: scores must be whole numbers
            ValueError: When scores refer to credits, credits must be within budget k
            ValueError: When scores refer to credits, credits must be perfect squares
            ValueError: When scores refer to votes, votes squared must be within budget k

        Returns: ScoreProfile where scores are the votes.
        """
        profile_copy = copy.deepcopy(profile)
        for b in profile_copy.ballots:
            # check to make sure that scores are whole numbers
            if b.scores is not None:
                if not all(isinstance(v, float) and v.is_integer() for v in b.scores.values()):
                    raise ValueError(f"Scores must be whole numbers.")
                if is_credits:  # is_credits -> scores mean credits
                    # check to make sure credits are in budget
                    if sum(b.scores.values()) > k:
                        raise ValueError(f"Ballot {b} is above the credit budget.")
                    # check to make sure credits are perfect squares
                    if not all(float(math.sqrt(v)).is_integer() for v in b.scores.values()):
                        raise ValueError(
                            f"Ballot {b} score violates credit's perfect squares requirement."
                        )
                    # convert scores to votes (takes the square root of score values)
                    # b.scores = {key: math.sqrt(v) for key, v in b.scores.items()}
                    for v in b.scores:
                        b.scores[v] = math.sqrt(b.scores[v])
                else:  # not is_credits -> scores mean votes
                    # convert scores (votes) to credits
                    # check to make sure credits are in budget
                    if sum(v**2 for v in b.scores.values()) > k:
                        raise ValueError(f"Ballot {b} is above the credit budget.")
        return profile_copy
