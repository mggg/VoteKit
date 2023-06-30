from pydantic import BaseModel
from typing import Optional

class Ballot(BaseModel):
    """
    id (optional string): assigned ballot id
    ranking (list): list of candidate ranking
    weight (float/fraction): weight assigned to a given a ballot
    voters (optional list): list of voters who cast a given a ballot
    """

    id: Optional[str] = None
    ranking: list[set]
    weight: float
    voters: Optional[set[str]] = None


# if __name__ == '__main__':
#     ballots1 = Ballot(id=None, ranking=[{'c'}, {np.nan}, {np.nan}], weight=1.0, voters={'a'})
#     ballots2 = Ballot(id=None, ranking=[{'c'}, {np.nan}, {np.nan}], weight=1.0, voters={'a'})
#     print(ballots1 == ballots2)
