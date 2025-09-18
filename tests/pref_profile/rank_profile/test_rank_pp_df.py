from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile
import pandas as pd

ballots_rankings = [
    RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
    RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
    RankBallot(),
    RankBallot(weight=0),
]


def test_pp_df_rankings():
    pp = RankProfile(ballots=ballots_rankings)
    data = {
        "Ranking_1": [
            frozenset({"A"}),
            frozenset({"A", "B"}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Ranking_2": [frozenset({"B"}), frozenset(), frozenset("~"), frozenset("~")],
        "Ranking_3": [
            frozenset({"C"}),
            frozenset({"D"}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [2.0, 1.0, 1.0, 0.0],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert pp.df.equals(true_df)


def test_pp_df_rankings_args():
    pp = RankProfile(
        ballots=ballots_rankings,
        candidates=["A", "B", "C", "D", "E"],
        max_ranking_length=4,
    )
    data = {
        "Ranking_1": [
            frozenset({"A"}),
            frozenset({"A", "B"}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Ranking_2": [frozenset({"B"}), frozenset(), frozenset("~"), frozenset("~")],
        "Ranking_3": [
            frozenset({"C"}),
            frozenset({"D"}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Ranking_4": [frozenset("~"), frozenset("~"), frozenset("~"), frozenset("~")],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [2.0, 1.0, 1.0, 0.0],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert pp.df.equals(true_df)
