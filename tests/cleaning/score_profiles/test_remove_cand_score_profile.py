from votekit.pref_profile import ScoreProfile, CleanedScoreProfile
from votekit.ballot import ScoreBallot
from votekit.cleaning import remove_cand_from_score_profile

profile = ScoreProfile(
    ballots=[
        ScoreBallot(scores={"A": 1, "B": 2, "C": 1.4}, weight=1),
        ScoreBallot(
            scores={
                "A": 1,
                "B": 2,
            },
            weight=1.3,
        ),
        ScoreBallot(
            scores={
                "D": 1,
                "C": 2,
            },
            weight=1,
        ),
    ]
)


def test_remove_cand():
    cleaned_profile = remove_cand_from_score_profile("A", profile)

    assert isinstance(cleaned_profile, CleanedScoreProfile)
    assert cleaned_profile.parent_profile == profile
    assert cleaned_profile.ballots == (
        ScoreBallot(scores={"B": 2, "C": 1.4}, weight=1),
        ScoreBallot(
            scores={
                "B": 2,
            },
            weight=1.3,
        ),
        ScoreBallot(
            scores={
                "D": 1,
                "C": 2,
            },
            weight=1,
        ),
    )

    assert cleaned_profile != profile
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_scores_altr_idxs == set()
    assert cleaned_profile.nonempty_altr_idxs == {0, 1}
    assert cleaned_profile.unaltr_idxs == {2}


def test_remove_mult_cands():
    cleaned_profile = remove_cand_from_score_profile(["A", "B"], profile)

    assert isinstance(cleaned_profile, CleanedScoreProfile)
    assert cleaned_profile.parent_profile == profile

    assert set(cleaned_profile.group_ballots().ballots) == set(
        [
            ScoreBallot(scores={"C": 1.4}, weight=1),
            ScoreBallot(
                scores={
                    "D": 1,
                    "C": 2,
                },
                weight=1,
            ),
        ]
    )
    assert cleaned_profile != profile
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_scores_altr_idxs == {1}
    assert cleaned_profile.nonempty_altr_idxs == {0}
    assert cleaned_profile.unaltr_idxs == {2}
