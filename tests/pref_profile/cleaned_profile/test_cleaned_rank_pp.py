from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from votekit.ballot import RankBallot
from votekit.pref_profile import (
    CleanedRankProfile,
    RankProfile,
)

profile = RankProfile(
    ballots=[
        RankBallot(ranking=[{"A"}, {"B"}], weight=1),
        RankBallot(weight=1),
        RankBallot(ranking=[{"C"}, {"B"}, {"A"}], weight=3),
        RankBallot(ranking=({"A"},)),
        RankBallot(ranking=({"B"},), weight=0),
    ]
)

clean_1 = CleanedRankProfile(
    ballots=[b for b in profile.ballots if b.weight > 0],
    parent_profile=profile,
    no_wt_altr_idxs={4},
    unaltr_idxs={0, 1, 2, 3},
    df_index_column=[0, 1, 2, 3],
)


clean_2 = CleanedRankProfile(
    ballots=[b for b in clean_1.ballots if b.ranking],
    parent_profile=clean_1,
    no_rank_altr_idxs={1},
    unaltr_idxs={0, 2, 3},
    df_index_column=[0, 2, 3],
)


def test_init():
    empty_profile = CleanedRankProfile(parent_profile=RankProfile(), df_index_column=[])

    true_df = pd.DataFrame(columns=["Voter Set", "Weight"], index=[], dtype=np.float64)
    true_df.index.name = "Ballot Index"

    assert empty_profile.ballots == ()
    assert not empty_profile.candidates
    assert empty_profile.to_df().equals(true_df)
    assert not empty_profile.candidates_cast
    assert not empty_profile.total_ballot_wt
    assert not empty_profile.num_ballots
    assert empty_profile.max_ranking_length == 0

    assert empty_profile.parent_profile == RankProfile()
    assert empty_profile.df_index_column == []
    assert empty_profile.no_wt_altr_idxs == set()
    assert empty_profile.no_rank_altr_idxs == set()
    assert empty_profile.nonempty_altr_idxs == set()
    assert empty_profile.unaltr_idxs == set()

    assert isinstance(empty_profile, RankProfile)


def test_parents():
    assert isinstance(clean_1, CleanedRankProfile)
    assert isinstance(clean_2, CleanedRankProfile)
    parent_profile = clean_2.parent_profile
    assert isinstance(parent_profile, CleanedRankProfile)
    assert clean_2.parent_profile == clean_1
    assert clean_1.parent_profile == profile
    assert parent_profile.parent_profile == profile


def test_reindexing_df():
    assert list(clean_1.to_df().index) == [0, 1, 2, 3]
    assert list(clean_2.to_df().index) == [0, 2, 3]


def test_no_wt_altr_idxs_subset_error():
    with pytest.raises(
        ValueError,
        match=("no_wt_altr_idxs is not a subset of the parent profile df index column."),
    ):
        CleanedRankProfile(
            ballots=[b for b in profile.ballots if b.weight > 0],
            parent_profile=profile,
            no_wt_altr_idxs={5},
            unaltr_idxs={0, 1, 2, 3},
            df_index_column=[0, 1, 2, 3],
        )


def test_no_ranking_altr_subset_error():
    with pytest.raises(
        ValueError,
        match=("no_rank_altr_idxs is not a subset of the parent profile df index column."),
    ):
        CleanedRankProfile(
            ballots=[b for b in profile.ballots if b.weight > 0],
            parent_profile=profile,
            no_wt_altr_idxs={4},
            no_rank_altr_idxs={5},
            unaltr_idxs={0, 1, 2, 3},
            df_index_column=[0, 1, 2, 3],
        )


def test_valid_but_alt_subset_error():
    with pytest.raises(
        ValueError,
        match=("nonempty_altr_idxs is not a subset of the parent profile df index column."),
    ):
        CleanedRankProfile(
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
        match=("unaltr_idxs is not a subset of the parent profile df index column."),
    ):
        CleanedRankProfile(
            ballots=[b for b in profile.ballots if b.weight > 0],
            parent_profile=profile,
            no_wt_altr_idxs={4},
            unaltr_idxs={0, 1, 2, 3, 5},
            df_index_column=[0, 1, 2, 3],
        )


def test_index_set_union_error():
    with pytest.raises(
        ValueError,
        match=("Union of ballot indices must equal the parent profile df index "),
    ):
        CleanedRankProfile(
            ballots=[b for b in profile.ballots if b.weight > 0],
            parent_profile=profile,
            no_wt_altr_idxs={4},
            unaltr_idxs={0, 1, 2},
            df_index_column=[0, 1, 2, 3],
        )


def test_group_ballots_warning():
    with pytest.warns(
        UserWarning,
        match="Grouping the ballots of a CleanedRankProfile will return a RankProfile",
    ):
        clean_1.group_ballots()


def test_cleaned_profile_str():
    assert "Profile has been cleaned" in str(clean_1)


def test_cleaned_profile_eq():
    clean = CleanedRankProfile(
        ballots=[b for b in profile.ballots if b.weight > 0],
        parent_profile=profile,
        no_wt_altr_idxs={4},
        unaltr_idxs={0, 1, 2, 3},
        df_index_column=[0, 1, 2, 3],
    )
    assert clean_1 == clean


@pytest.mark.parametrize("from_df", [False, True], ids=["ballots", "dataframe"])
def test_constructor_owns_metadata_and_to_df_returns_detached_snapshots(from_df: bool) -> None:
    parent = clean_1
    retained_df = clean_2.to_df()
    indices = list(retained_df.index)
    no_weight = set(clean_2.no_wt_altr_idxs)
    no_payload = set(clean_2.no_rank_altr_idxs)
    altered = set(clean_2.nonempty_altr_idxs)
    unaltered = set(clean_2.unaltr_idxs)
    cleaned = CleanedRankProfile(
        parent_profile=parent,
        df_index_column=indices,
        no_wt_altr_idxs=no_weight,
        no_rank_altr_idxs=no_payload,
        nonempty_altr_idxs=altered,
        unaltr_idxs=unaltered,
        candidates=clean_2.candidates,
        max_ranking_length=clean_2.max_ranking_length,
        df=retained_df if from_df else pd.DataFrame(),
        ballots=() if from_df else clean_2.ballots,
    )
    expected_df = cleaned.to_df()
    expected_parent_df = parent.to_df()
    expected_ballots = cleaned.ballots
    expected_total = cleaned.total_ballot_wt

    indices[:] = [99]
    no_weight.add(99)
    no_payload.add(99)
    altered.add(99)
    unaltered.add(99)

    assert cleaned.df_index_column == clean_2.df_index_column
    assert cleaned.no_wt_altr_idxs == clean_2.no_wt_altr_idxs
    assert cleaned.no_rank_altr_idxs == clean_2.no_rank_altr_idxs
    assert cleaned.nonempty_altr_idxs == clean_2.nonempty_altr_idxs
    assert cleaned.unaltr_idxs == clean_2.unaltr_idxs

    exported_df = cleaned.to_df()
    first_index = exported_df.index[0]
    exported_df["Ranking_1"] = pd.Series(
        [frozenset({"B"})] * len(exported_df), index=exported_df.index, dtype=object
    )
    exported_df.at[first_index, "Weight"] = 99.0
    exported_df["Voter Set"] = pd.Series(
        [frozenset({"other voter"})] * len(exported_df), index=exported_df.index, dtype=object
    )

    assert not np.shares_memory(
        exported_df.index.to_numpy(), cleaned._matrix.voter_weight_data.ballot_index
    )

    exported_df.rename(index={first_index: 99}, inplace=True)
    exported_df.index.name = "Changed index"

    pd.testing.assert_frame_equal(cleaned.to_df(), expected_df)
    pd.testing.assert_frame_equal(cleaned._matrix.to_df(), expected_df)
    pd.testing.assert_frame_equal(cleaned.copy().to_df(), expected_df)
    pd.testing.assert_frame_equal(parent.to_df(), expected_parent_df)
    assert cleaned.ballots == expected_ballots
    assert cleaned.total_ballot_wt == expected_total


@pytest.mark.parametrize("from_df", [False, True], ids=["ballots", "dataframe"])
def test_grouping_returns_ordinary_profile_with_reset_indices(from_df: bool) -> None:
    retained_df = clean_2.to_df()
    retained_df.index = pd.Index(retained_df.index + 10, name="Ballot Index")
    parent = RankProfile(
        df=retained_df,
        candidates=clean_2.candidates,
        max_ranking_length=clean_2.max_ranking_length,
    )
    cleaned = CleanedRankProfile(
        parent_profile=parent,
        df_index_column=list(retained_df.index),
        unaltr_idxs=set(retained_df.index),
        candidates=clean_2.candidates,
        max_ranking_length=clean_2.max_ranking_length,
        df=retained_df if from_df else pd.DataFrame(),
        ballots=() if from_df else clean_2.ballots,
    )

    with pytest.warns(UserWarning, match="resets ballot indices"):
        grouped = cleaned.group_ballots()

    assert type(grouped) is RankProfile
    assert grouped.to_df().index.tolist() == list(range(grouped.num_ballots))
    assert grouped.to_df().index.name == "Ballot Index"
    assert grouped.total_ballot_wt == cleaned.total_ballot_wt
    assert grouped == RankProfile(ballots=cleaned.ballots).group_ballots()
    pd.testing.assert_frame_equal(cleaned.to_df(), retained_df)


@pytest.mark.parametrize("labels", [[0, 1], [7, 7], [10, 20]])
def test_cleaned_rank_validation_reads_parent_labels(
    labels: list[int], forbid_dataframe_conversion: Callable[[], None]
) -> None:
    ballot = RankBallot(ranking=[{"A"}])
    source_df = RankProfile(ballots=[ballot, ballot]).to_df()
    source_df.index = pd.Index(labels, name="Ballot Index")
    parent = RankProfile(df=source_df, candidates=("A",), max_ranking_length=1)

    forbid_dataframe_conversion()

    cleaned = CleanedRankProfile(
        parent_profile=parent,
        ballots=[ballot, ballot],
        df_index_column=labels,
        unaltr_idxs=set(labels),
    )

    assert cleaned.parent_profile is parent

    stored_labels = cleaned._matrix.voter_weight_data.ballot_index

    assert stored_labels is not None
    assert stored_labels.tolist() == labels

    with pytest.raises(ValueError, match="not a subset"):
        CleanedRankProfile(parent_profile=parent, df_index_column=[], unaltr_idxs={99})
