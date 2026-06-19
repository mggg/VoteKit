import numpy as np
import pandas as pd

from votekit.ballot import ScoreBallot
from votekit.pref_profile import ScoreProfile

ballots_scores = [
    ScoreBallot(
        weight=2,
        scores={
            "A": 1,
            "B": 2,
        },
    ),
    ScoreBallot(scores={"D": 2, "E": 1}, voter_set={"Chris"}),
    ScoreBallot(),
    ScoreBallot(weight=0),
]

mixed_ballots_scores = [
    ScoreBallot(
        weight=2,
        scores={
            "A": 1,
            1: 2,
        },
    ),
    ScoreBallot(scores={"A": 2, "B": 1}, voter_set={"Chris"}),
    ScoreBallot(scores={1: 2, 2: 1}),
    ScoreBallot(),
    ScoreBallot(weight=0),
]


def test_pp_df_scores():
    pp = ScoreProfile(ballots=ballots_scores)
    data = {
        "A": [
            1,
            np.nan,
            np.nan,
            np.nan,
        ],
        "B": [
            2,
            np.nan,
            np.nan,
            np.nan,
        ],
        "D": [np.nan, 2, np.nan, np.nan],
        "E": [np.nan, 1, np.nan, np.nan],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [2.0, 1.0, 1.0, 0.0],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert pp.df.equals(true_df)


def test_pp_df_scores_args():
    pp = ScoreProfile(
        ballots=ballots_scores,
        candidates=["A", "B", "C", "D", "E"],
    )
    data = {
        "A": [
            1,
            np.nan,
            np.nan,
            np.nan,
        ],
        "B": [
            2,
            np.nan,
            np.nan,
            np.nan,
        ],
        "C": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        "D": [np.nan, 2, np.nan, np.nan],
        "E": [np.nan, 1, np.nan, np.nan],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [2.0, 1.0, 1.0, 0.0],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert pp.df.equals(true_df)


def test_df_with_mixed_cand_types_as_score_cols():
    score_profile = ScoreProfile(ballots=mixed_ballots_scores, candidates=["A", "B", "C", 1, 2, 3])

    data = {
        "A": [
            1,
            2,
            np.nan,
            np.nan,
            np.nan,
        ],
        "B": [
            np.nan,
            1,
            np.nan,
            np.nan,
            np.nan,
        ],
        "C": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        1: [2, np.nan, 2, np.nan, np.nan],
        2: [np.nan, np.nan, 1, np.nan, np.nan],
        3: [np.nan, np.nan, np.nan, np.nan, np.nan],
        "Voter Set": [set(), {"Chris"}, set(), set(), set()],
        "Weight": [2.0, 1.0, 1.0, 1.0, 0.0],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert score_profile.df.equals(true_df)


def test_internal_df_with_cand_ids_as_score_cols():
    score_profile = ScoreProfile(
        ballots=ballots_scores,
        candidates=["A", "B", "C", "D", "E"],
    )
    candidate_ids = set([i for i in range(len(score_profile.candidates))])
    candidate_id_map = dict(zip(score_profile.candidates, candidate_ids))
    candidates_cast_ids = set([candidate_id_map[cand] for cand in score_profile.candidates_cast])

    id_A = candidate_id_map["A"]
    id_B = candidate_id_map["B"]
    id_C = candidate_id_map["C"]
    id_D = candidate_id_map["D"]
    id_E = candidate_id_map["E"]
    cand_id_data = {
        id_A: [
            1,
            np.nan,
            np.nan,
            np.nan,
        ],
        id_B: [
            2,
            np.nan,
            np.nan,
            np.nan,
        ],
        id_C: [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        id_D: [np.nan, 2, np.nan, np.nan],
        id_E: [np.nan, 1, np.nan, np.nan],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [2.0, 1.0, 1.0, 0.0],
    }
    true_id_df = pd.DataFrame(cand_id_data)
    true_id_df.index.name = "Ballot Index"
    assert score_profile._df.equals(true_id_df)
    assert score_profile._candidates == tuple(candidate_ids)
    assert score_profile._candidates_cast == tuple(candidates_cast_ids)
    assert score_profile.candidate_id_map == candidate_id_map
