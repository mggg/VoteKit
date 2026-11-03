from votekit.ballot import ScoreBallot
from votekit.cleaning import remove_cand_score_ballot


def test_remove_candidate_preserves_other_explicit_zero_scores():
    ballot = ScoreBallot(scores={"A": 1, "B": 0}, include_zero_score=True)

    cleaned = remove_cand_score_ballot("A", ballot)

    assert cleaned.scores == {"B": 0.0}
    assert ballot.scores == {"A": 1.0, "B": 0.0}
    assert remove_cand_score_ballot("B", ballot).scores == {"A": 1.0}


def test_remove_cand_score_ballot_singular():
    b = ScoreBallot(scores={"A": 1, "B": 2}, weight=2.1, voter_set={"Chris"})

    rc = remove_cand_score_ballot("A", b)

    assert rc.scores == {"B": 2}
    assert rc.weight == 2.1
    assert rc.voter_set == {"Chris"}


def test_remove_cand_score_ballot_plural():
    b = ScoreBallot(scores={"A": 1, "B": 2, "C": 2.1}, weight=2.1, voter_set={"Chris"})

    rc = remove_cand_score_ballot(["A", "B"], b)

    assert rc.scores == {"C": 2.1}
    assert rc.weight == 2.1
    assert rc.voter_set == {"Chris"}


def test_remove_cand_score_ballot_all():
    b = ScoreBallot(scores={"A": 1, "B": 2}, weight=2.1, voter_set={"Chris"})

    rc = remove_cand_score_ballot(["A", "B"], b)

    assert rc.scores is None
    assert rc.weight == 2.1
    assert rc.voter_set == {"Chris"}
