from typing import cast

import numpy as np
import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.matrices import candidate_distance


def test_candidate_distance():
    b = RankBallot(ranking=({"Chris"}, {"Peter", "Moon"}, {"Jeanne"}))

    assert candidate_distance("Chris", "Peter", b) == 1
    assert candidate_distance("Chris", "Moon", b) == 1
    assert candidate_distance("Moon", "Peter", b) == 0
    assert candidate_distance("Chris", "Jeanne", b) == 2
    assert candidate_distance("Jeanne", "Chris", b) == -2


def test_candidate_distance_no_ranking_error():
    b = ScoreBallot(scores={"Chris": 4})

    with pytest.raises(TypeError, match="Ballot must be of type RankBallot"):
        candidate_distance("Chris", "Peter", cast(RankBallot, b))


def test_candidate_distance_nan_candidate():
    b = RankBallot(ranking=({"Chris"}, {"Peter", "Moon"}, {"Jeanne"}))

    assert np.isnan(candidate_distance("Chris", "Mala", b))
    assert np.isnan(candidate_distance("Mala", "Mala", b))
