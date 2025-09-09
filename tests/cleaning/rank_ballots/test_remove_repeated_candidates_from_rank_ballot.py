import pytest
from votekit.ballot import RankBallot, ScoreBallot
from votekit.cleaning import remove_repeated_cands_from_rank_ballot


def test_remove_repeated_cands():
    b = RankBallot(
        ranking=[{"A"}, {"A"}, {"B"}, {"C"}, {"B"}], weight=2.1, voter_set={"Chris"}
    )
    rrc = remove_repeated_cands_from_rank_ballot(b)

    assert rrc.weight == 2.1
    assert rrc.voter_set == {"Chris"}
    assert rrc.ranking == ({"A"}, set(), {"B"}, {"C"}, set())


def test_errors():
    with pytest.raises(TypeError, match="Ballot must have rankings:"):
        remove_repeated_cands_from_rank_ballot(RankBallot())

    with pytest.raises(TypeError, match="Ballot must be of type RankBallot."):
        remove_repeated_cands_from_rank_ballot(ScoreBallot())
