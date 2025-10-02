from votekit.pref_profile import ScoreProfile, CleanedScoreProfile
from votekit.ballot import ScoreBallot
from votekit.cleaning import clean_score_profile
import numpy as np

profile = ScoreProfile(
    ballots=[
        ScoreBallot(scores={"A": 2, "B": 2.1}, weight=1),
        ScoreBallot(scores={"A": 2, "B": 2}, weight=1),
        ScoreBallot(scores={"A": 1.1, "B": 1.23}, weight=1),
        ScoreBallot(scores={"B": 1}, weight=0),
    ]
)


def test_clean_profile_with_defaults():
    adj_profile = clean_score_profile(
        profile,
        lambda x: tuple(
            np.nan if not np.isnan(score) and int(score) != score else score
            for score in x
        ),
    )

    assert isinstance(adj_profile, CleanedScoreProfile)
    assert adj_profile.parent_profile == profile
    assert adj_profile.ballots == (
        ScoreBallot(scores={"A": 2}, weight=1),
        ScoreBallot(scores={"A": 2, "B": 2}, weight=1),
    )
    assert adj_profile != profile

    assert adj_profile.no_wt_altr_idxs == set()
    assert adj_profile.no_scores_altr_idxs == {2}
    assert adj_profile.nonempty_altr_idxs == {0}
    assert adj_profile.unaltr_idxs == {1, 3}


def test_clean_profile_change_defaults():
    adj_profile = clean_score_profile(
        profile,
        lambda x: tuple(
            np.nan if not np.isnan(score) and int(score) != score else score
            for score in x
        ),
        remove_empty_ballots=False,
        remove_zero_weight_ballots=False,
        retain_original_candidate_list=True,
    )
    assert isinstance(adj_profile, CleanedScoreProfile)
    assert adj_profile.parent_profile == profile
    assert set(adj_profile.ballots) == set(
        (
            ScoreBallot(scores={"A": 2}, weight=1),
            ScoreBallot(scores={"A": 2, "B": 2}, weight=1),
            ScoreBallot(weight=1),
            ScoreBallot(scores={"B": 1}, weight=0),
        )
    )

    assert adj_profile.candidates == profile.candidates
    assert adj_profile.no_wt_altr_idxs == set()
    assert adj_profile.no_scores_altr_idxs == {2}
    assert adj_profile.nonempty_altr_idxs == {0}
    assert adj_profile.unaltr_idxs == {1, 3}
