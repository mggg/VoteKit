from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile


ballots = [
    RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
    RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
    RankBallot(),
    RankBallot(weight=0),
]


def test_print_profile_rankings():
    print(RankProfile(ballots=ballots))

    assert RankProfile(ballots=ballots).__str__() == (
        f"RankProfile\n"
        f"Maximum ranking length: 3\n"
        f"Candidates: {('A', 'B', 'C', 'D')}\n"
        f"Candidates who received votes: {('A', 'B', 'C', 'D')}\n"
        f"Total number of Ballot objects: 4\n"
        f"Total weight of Ballot objects: 4.0\n"
    )
