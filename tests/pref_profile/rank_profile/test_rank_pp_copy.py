from fractions import Fraction

import pandas as pd
import pytest

from votekit.ballot import RankBallot
from votekit.cleaning import remove_and_condense_rank_profile
from votekit.pref_profile import RankProfile


@pytest.mark.parametrize("weight", [2.0, Fraction(2, 3)])
@pytest.mark.parametrize("cleaned", [False, True])
def test_copy_shares_storage_but_preserves_independent_profile_and_exports(
    weight, cleaned, monkeypatch
):
    source_df = RankProfile(
        ballots=[
            RankBallot(ranking=[{"A", 1}, {"B"}], weight=weight, voter_set={"voter"}),
            RankBallot(ranking=[{1}], weight=0),
        ],
        candidates=("B", "unused", 1, "A"),
        max_ranking_length=3,
    ).to_df()
    source_df.index = pd.Index([17, 17], name="Ballot Index")
    profile = RankProfile(df=source_df, candidates=("B", "unused", 1, "A"), max_ranking_length=3)
    if cleaned:
        profile = remove_and_condense_rank_profile("A", profile, remove_zero_weight_ballots=False)
    expected_df = profile.to_df()
    expected_ballots = profile.ballots

    def reject_round_trip(*args, **kwargs):
        raise AssertionError(
            "Copy must not export or reconstruct a profile through its public constructor"
        )

    with monkeypatch.context() as patch:
        patch.setattr(RankProfile, "to_df", reject_round_trip)
        patch.setattr(RankProfile, "__init__", reject_round_trip)
        copied = profile.copy()

    assert type(copied) is RankProfile
    assert copied is not profile
    assert copied._matrix is profile._matrix
    assert not hasattr(copied, "parent_profile")
    assert copied.ballots == expected_ballots
    assert copied.candidates == profile.candidates
    assert set(copied.candidates_cast) == set(profile.candidates_cast)
    assert copied.max_ranking_length == profile.max_ranking_length
    assert copied.total_ballot_wt == profile.total_ballot_wt
    assert copied.max_candidates_ranked == profile.max_candidates_ranked
    pd.testing.assert_frame_equal(copied.to_df(), expected_df)

    exported = copied.to_df()
    exported["Weight"] = 99.0
    exported["Ranking_1"] = pd.Series([frozenset({"unused"})] * len(exported), index=exported.index)
    pd.testing.assert_frame_equal(profile.to_df(), expected_df)
    pd.testing.assert_frame_equal(copied.to_df(), expected_df)
    with pytest.raises(AttributeError, match="Cannot modify frozen instance"):
        copied.candidates = ()


def test_empty_subclass_copy_returns_ordinary_profile():
    class SpecializedProfile(RankProfile):
        pass

    profile = SpecializedProfile(candidates=("unused",), max_ranking_length=3)
    copied = profile.copy()
    assert type(copied) is RankProfile
    assert copied._matrix is profile._matrix
    assert copied.num_ballots == 0
    assert copied.total_ballot_wt == 0
    assert copied.candidates == ("unused",)
    assert copied.max_ranking_length == 3
    pd.testing.assert_frame_equal(copied.to_df(), profile.to_df())
