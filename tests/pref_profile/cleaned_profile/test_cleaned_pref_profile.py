from votekit.pref_profile import PreferenceProfile, CleanedProfile
from votekit.ballot import Ballot
import pandas as pd
import pytest
import numpy as np

profile = PreferenceProfile(
    ballots=[
        Ballot(ranking=[{"A"}, {"B"}], weight=1),
        Ballot(weight=1),
        Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=3),
        Ballot(ranking=({"A"},)),
        Ballot(ranking=({"B"},), weight=0),
    ]
)

clean_1 = CleanedProfile(
    ballots=[b for b in profile.ballots if b.weight > 0],
    parent_profile=profile,
    no_wt_altr_idxs={4},
    unaltr_idxs={0, 1, 2, 3},
    df_index_column=[0, 1, 2, 3],
)

clean_2 = CleanedProfile(
    ballots=[b for b in clean_1.ballots if b.ranking or b.scores],
    parent_profile=clean_1,
    no_rank_no_score_altr_idxs={1},
    unaltr_idxs={0, 2, 3},
    df_index_column=[0, 2, 3],
)


def test_init():
    empty_profile = CleanedProfile(
        parent_profile=PreferenceProfile(), df_index_column=[]
    )

    true_df = pd.DataFrame(columns=["Voter Set", "Weight"], index=[], dtype=np.float64)
    true_df.index.name = "Ballot Index"

    assert empty_profile.ballots == ()
    assert not empty_profile.candidates
    assert empty_profile.df.equals(true_df)
    assert not empty_profile.candidates_cast
    assert not empty_profile.total_ballot_wt
    assert not empty_profile.num_ballots
    assert empty_profile.max_ranking_length == 0

    assert empty_profile.parent_profile == PreferenceProfile()
    assert empty_profile.df_index_column == []
    assert empty_profile.no_wt_altr_idxs == set()
    assert empty_profile.no_rank_no_score_altr_idxs == set()
    assert empty_profile.nonempty_altr_idxs == set()
    assert empty_profile.unaltr_idxs == set()


def test_parents():
    assert clean_2.parent_profile == clean_1
    assert clean_1.parent_profile == profile
    assert clean_2.parent_profile.parent_profile == profile


def test_reindexing_df():
    assert list(clean_1.df.index) == [0, 1, 2, 3]
    assert list(clean_2.df.index) == [0, 2, 3]


def test_no_wt_altr_idxs_subset_error():
    with pytest.raises(
        ValueError,
        match=(
            "no_wt_altr_idxs is not a subset of " "the parent profile df index column."
        ),
    ):
        CleanedProfile(
            ballots=[b for b in profile.ballots if b.weight > 0],
            parent_profile=profile,
            no_wt_altr_idxs={5},
            unaltr_idxs={0, 1, 2, 3},
            df_index_column=[0, 1, 2, 3],
        )


def test_no_ranking_and_no_scores_altr_subset_error():
    with pytest.raises(
        ValueError,
        match=(
            "no_rank_no_score_altr_idxs is not a subset of "
            "the parent profile df index column."
        ),
    ):
        CleanedProfile(
            ballots=[b for b in profile.ballots if b.weight > 0],
            parent_profile=profile,
            no_wt_altr_idxs={4},
            no_rank_no_score_altr_idxs={5},
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
        CleanedProfile(
            ballots=[b for b in profile.ballots if b.weight > 0],
            parent_profile=profile,
            no_wt_altr_idxs={4},
            nonempty_altr_idxs={5},
            unaltr_idxs={0, 1, 2, 3},
            df_index_column=[0, 1, 2, 3],
        )


def test_unaltr_subset_error():
    with pytest.raises(
        ValueError,
        match=("unaltr_idxs is not a subset of " "the parent profile df index column."),
    ):
        CleanedProfile(
            ballots=[b for b in profile.ballots if b.weight > 0],
            parent_profile=profile,
            no_wt_altr_idxs={4},
            unaltr_idxs={0, 1, 2, 3, 5},
            df_index_column=[0, 1, 2, 3],
        )


def test_index_set_union_error():
    with pytest.raises(
        ValueError,
        match=("Union of ballot indices must equal the parent profile " "df index "),
    ):
        CleanedProfile(
            ballots=[b for b in profile.ballots if b.weight > 0],
            parent_profile=profile,
            no_wt_altr_idxs={4},
            unaltr_idxs={0, 1, 2},
            df_index_column=[0, 1, 2, 3],
        )


def test_group_ballots_warning():
    with pytest.warns(
        UserWarning,
        match="Grouping the ballots of a CleanedProfile will return a PreferenceProfile",
    ):
        clean_1.group_ballots()


def test_cleaned_profile_str():
    assert "Profile has been cleaned" in str(clean_1)


def test_cleaned_profile_eq():
    clean = CleanedProfile(
        ballots=[b for b in profile.ballots if b.weight > 0],
        parent_profile=profile,
        no_wt_altr_idxs={4},
        unaltr_idxs={0, 1, 2, 3},
        df_index_column=[0, 1, 2, 3],
    )
    assert clean_1 == clean
