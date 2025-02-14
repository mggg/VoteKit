from votekit.ballot import Ballot
from votekit.matrices import candidate_distance
import pytest
import numpy as np


def test_candidate_distance():
    b = Ballot(ranking=({"Chris"}, {"Peter", "Moon"}, {"Jeanne"}))

    assert candidate_distance("Chris", "Peter", b) == 1
    assert candidate_distance("Chris", "Moon", b) == 1
    assert candidate_distance("Moon", "Peter", b) == 0
    assert candidate_distance("Chris", "Jeanne", b) == 2
    assert candidate_distance("Jeanne", "Chris", b) == -2


def test_candidate_distance_no_ranking_error():
    b = Ballot(scores={"Chris": 4})

    with pytest.raises(TypeError, match="Ballot must have a ranking."):
        candidate_distance("Chris", "Peter", b)


def test_candidate_distance_nan_candidate():
    b = Ballot(ranking=({"Chris"}, {"Peter", "Moon"}, {"Jeanne"}))

    assert np.isnan(candidate_distance("Chris", "Mala", b))
    assert np.isnan(candidate_distance("Mala", "Mala", b))
