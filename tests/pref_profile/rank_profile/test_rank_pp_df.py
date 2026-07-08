import pandas as pd

from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile

ballots_rankings = [
    RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
    RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
    RankBallot(),
    RankBallot(weight=0),
]
mixed_ballot_rankings = [
    RankBallot(ranking=({"A"}, {1}, {2}), weight=2),
    RankBallot(ranking=({"A", "B"}, frozenset(), {"C"}), voter_set={"Chris"}),
    RankBallot(ranking=(2, 1)),
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


def test_df_with_mixed_cand_types_as_ranking_values():
    rank_profile = RankProfile(
        ballots=mixed_ballot_rankings,
        candidates=["A", "B", "C", 1, 2, 3],
        max_ranking_length=4,
    )

    data = {
        "Ranking_1": [
            frozenset({"A"}),
            frozenset({"A", "B"}),
            frozenset({2}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Ranking_2": [frozenset({1}), frozenset(), frozenset({1}), frozenset("~"), frozenset("~")],
        "Ranking_3": [
            frozenset({2}),
            frozenset({"C"}),
            frozenset("~"),
            frozenset("~"),
            frozenset("~"),
        ],
        "Ranking_4": [
            frozenset("~"),
            frozenset("~"),
            frozenset("~"),
            frozenset("~"),
            frozenset("~"),
        ],
        "Voter Set": [set(), {"Chris"}, set(), set(), set()],
        "Weight": [2.0, 1.0, 1.0, 1.0, 0.0],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert rank_profile.df.equals(true_df)


def test_internal_df_with_cand_ids_as_ranking_values():
    rank_profile = RankProfile(
        ballots=ballots_rankings,
        candidates=["A", "B", "C", "D", "E"],
        max_ranking_length=4,
    )
    candidate_id_map = rank_profile.candidate_id_map

    id_A = candidate_id_map[frozenset({"A"})]
    id_B = candidate_id_map[frozenset({"B"})]
    id_C = candidate_id_map[frozenset({"C"})]
    id_D = candidate_id_map[frozenset({"D"})]
    id_AB_tie = candidate_id_map[frozenset({"A", "B"})]
    id_tilda = candidate_id_map[frozenset({"~"})]
    id_empty = candidate_id_map[frozenset()]
    cand_id_data = {
        "Ranking_1": [
            id_A,
            id_AB_tie,
            id_tilda,
            id_tilda,
        ],
        "Ranking_2": [id_B, id_empty, id_tilda, id_tilda],
        "Ranking_3": [
            id_C,
            id_D,
            id_tilda,
            id_tilda,
        ],
        "Ranking_4": [id_tilda, id_tilda, id_tilda, id_tilda],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [2.0, 1.0, 1.0, 0.0],
    }
    true_id_df = pd.DataFrame(cand_id_data)
    true_id_df.index.name = "Ballot Index"
    assert rank_profile._df.equals(true_id_df)
