from votekit.pref_profile import CleanedScoreProfile, ScoreProfile
from votekit.ballot import ScoreBallot
import pandas as pd
import pytest
import numpy as np

profile = ScoreProfile(
    ballots=[
        ScoreBallot(scores={"A": 2, "B": 2.1}, weight=1),
        ScoreBallot(weight=1),
        ScoreBallot(scores={"A": 1}, weight=0),
    ]
)

clean_1 = CleanedScoreProfile(
    ballots=[b for b in profile.ballots if b.weight > 0],
    parent_profile=profile,
    no_wt_altr_idxs={2},
    unaltr_idxs={
        0,
        1,
    },
    df_index_column=[0, 1],
)

clean_2 = CleanedScoreProfile(
    ballots=[b for b in clean_1.ballots if b.scores],
    parent_profile=clean_1,
    no_scores_altr_idxs={1},
    unaltr_idxs={0},
    df_index_column=[0],
)


def test_init():
    empty_profile = CleanedScoreProfile(
        parent_profile=ScoreProfile(), df_index_column=[]
    )

    true_df = pd.DataFrame(columns=["Voter Set", "Weight"], index=[], dtype=np.float64)
    true_df.index.name = "Ballot Index"

    assert empty_profile.ballots == ()
    assert not empty_profile.candidates
    assert empty_profile.df.equals(true_df)
    assert not empty_profile.candidates_cast
    assert not empty_profile.total_ballot_wt
    assert not empty_profile.num_ballots

    assert empty_profile.parent_profile == ScoreProfile()
    assert empty_profile.df_index_column == []
    assert empty_profile.no_wt_altr_idxs == set()
    assert empty_profile.no_scores_altr_idxs == set()
    assert empty_profile.nonempty_altr_idxs == set()
    assert empty_profile.unaltr_idxs == set()


def test_parents():
    assert clean_2.parent_profile == clean_1
    assert clean_1.parent_profile == profile
    assert clean_2.parent_profile.parent_profile == profile


def test_reindexing_df():
    assert list(clean_1.df.index) == [0, 1]
    assert list(clean_2.df.index) == [0]


def test_no_wt_altr_idxs_subset_error():
    with pytest.raises(
        ValueError,
        match=(
            "no_wt_altr_idxs is not a subset of " "the parent profile df index column."
        ),
    ):
        CleanedScoreProfile(
            ballots=[b for b in profile.ballots if b.weight > 0],
            parent_profile=profile,
            no_wt_altr_idxs={5},
            unaltr_idxs={0, 1, 2, 3},
            df_index_column=[0, 1, 2, 3],
        )


def test_no_scores_altr_subset_error():
    with pytest.raises(
        ValueError,
        match=(
            "no_scores_altr_idxs is not a subset of "
            "the parent profile df index column."
        ),
    ):
        CleanedScoreProfile(
            ballots=[b for b in profile.ballots if b.weight > 0],
            parent_profile=profile,
            no_wt_altr_idxs={2},
            no_scores_altr_idxs={5},
            unaltr_idxs={0, 1, 2, 3},
            df_index_column=[0, 1, 2, 3],
        )


def test_valid_but_alt_subset_error():
    with pytest.raises(
        ValueError,
        match=(
            "nonempty_altr_idxs is not a subset of "
            "the parent profile df index column."
        ),
    ):
        CleanedScoreProfile(
            ballots=[b for b in profile.ballots if b.weight > 0],
            parent_profile=profile,
            no_wt_altr_idxs={2},
            nonempty_altr_idxs={5},
            unaltr_idxs={0, 1, 2, 3},
            df_index_column=[0, 1, 2, 3],
        )


def test_unaltr_subset_error():
    with pytest.raises(
        ValueError,
        match=("unaltr_idxs is not a subset of " "the parent profile df index column."),
    ):
        CleanedScoreProfile(
            ballots=[b for b in profile.ballots if b.weight > 0],
            parent_profile=profile,
            no_wt_altr_idxs={2},
            unaltr_idxs={0, 1, 2, 3, 5},
            df_index_column=[0, 1, 2, 3],
        )


def test_index_set_union_error():
    with pytest.raises(
        ValueError,
        match=("Union of ballot indices must equal the parent profile " "df index "),
    ):
        CleanedScoreProfile(
            ballots=[b for b in profile.ballots if b.weight > 0],
            parent_profile=profile,
            no_wt_altr_idxs={2},
            unaltr_idxs={
                0,
            },
            df_index_column=[0, 1],
        )


def test_group_ballots_warning():
    with pytest.warns(
        UserWarning,
        match="Grouping the ballots of a CleanedScoreProfile will return a ScoreProfile",
    ):
        clean_1.group_ballots()


def test_cleaned_profile_str():
    assert "Profile has been cleaned" in str(clean_1)


def test_cleaned_profile_eq():
    clean = CleanedScoreProfile(
        ballots=[b for b in profile.ballots if b.weight > 0],
        parent_profile=profile,
        no_wt_altr_idxs={2},
        unaltr_idxs={0, 1},
        df_index_column=[
            0,
            1,
        ],
    )
    assert clean_1 == clean
