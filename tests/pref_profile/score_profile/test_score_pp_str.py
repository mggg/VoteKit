from votekit.ballot import ScoreBallot
from votekit.pref_profile import ScoreProfile


ballots = [
    ScoreBallot(scores={"A": 2}, weight=2),
    ScoreBallot(scores={"B": 2}, voter_set={"Chris"}),
    ScoreBallot(),
    ScoreBallot(weight=0),
]

p = ScoreProfile(ballots=ballots, candidates=["A", "B", "C", "D"])


def test_print_profile_rankings():
    print(p)

    assert p.__str__() == (
        f"ScoreProfile\n"
        f"Candidates: {('A', 'B', 'C', 'D')}\n"
        f"Candidates who received votes: {('A', 'B')}\n"
        f"Total number of Ballot objects: 4\n"
        f"Total weight of Ballot objects: 4.0\n"
    )
