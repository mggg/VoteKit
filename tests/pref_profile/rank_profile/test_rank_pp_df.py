import pandas as pd

from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile

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


def test_internal_df_with_cand_ids_as_ranking_values():
    rank_profile = RankProfile(
        ballots=ballots_rankings,
        candidates=["A", "B", "C", "D", "E"],
        max_ranking_length=4,
    )
    candidate_ids = set([i for i in range(len(rank_profile.candidates))])
    candidate_id_map = dict(zip(rank_profile.candidates, candidate_ids))
    candidates_cast_ids = set([candidate_id_map[cand] for cand in rank_profile.candidates_cast])

    id_A = candidate_id_map["A"]
    id_B = candidate_id_map["B"]
    id_C = candidate_id_map["C"]
    id_D = candidate_id_map["D"]
    cand_id_data = {
        "Ranking_1": [
            frozenset({id_A}),
            frozenset({id_A, id_B}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Ranking_2": [frozenset({id_B}), frozenset(), frozenset("~"), frozenset("~")],
        "Ranking_3": [
            frozenset({id_C}),
            frozenset({id_D}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Ranking_4": [frozenset("~"), frozenset("~"), frozenset("~"), frozenset("~")],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [2.0, 1.0, 1.0, 0.0],
    }
    true_id_df = pd.DataFrame(cand_id_data)
    true_id_df.index.name = "Ballot Index"
    assert rank_profile._df.equals(true_id_df)
    assert rank_profile._candidates == tuple(candidate_ids)
    assert rank_profile._candidates_cast == tuple(candidates_cast_ids)
    assert rank_profile.candidate_id_map == candidate_id_map
