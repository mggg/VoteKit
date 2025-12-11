from votekit.ballot import ScoreBallot
from votekit.pref_profile import ScoreProfile


def test_pp_total_ballot_wt():

    pp = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2, "B": 2}, weight=2),
            ScoreBallot(scores={"A": 2, "B": 2}, voter_set={"Chris"}),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
    )
    assert pp.total_ballot_wt == 4

    pp = pp.group_ballots()
    assert pp.total_ballot_wt == 4
