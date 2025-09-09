from votekit.ballot import ScoreBallot
from votekit.cleaning import remove_cand_from_score_ballot


def test_remove_cand_from_score_ballot_singular():
    b = ScoreBallot(scores={"A": 1, "B": 2}, weight=2.1, voter_set={"Chris"})

    rc = remove_cand_from_score_ballot("A", b)

    assert rc.scores == {"B": 2}
    assert rc.weight == 2.1
    assert rc.voter_set == {"Chris"}


def test_remove_cand_from_score_ballot_plural():
    b = ScoreBallot(scores={"A": 1, "B": 2, "C": 2.1}, weight=2.1, voter_set={"Chris"})

    rc = remove_cand_from_score_ballot(["A", "B"], b)

    assert rc.scores == {"C": 2.1}
    assert rc.weight == 2.1
    assert rc.voter_set == {"Chris"}


def test_remove_cand_from_score_ballot_all():
    b = ScoreBallot(scores={"A": 1, "B": 2}, weight=2.1, voter_set={"Chris"})

    rc = remove_cand_from_score_ballot(["A", "B"], b)

    assert rc.scores is None
    assert rc.weight == 2.1
    assert rc.voter_set == {"Chris"}
