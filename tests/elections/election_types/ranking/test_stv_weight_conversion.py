from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from votekit.ballot import RankBallot
from votekit.cleaning import remove_and_condense_rank_profile
from votekit.elections.election_types.ranking.stv.exact_fraction_ops import (
    convert_profile_weights,
    to_exact_fraction_weight,
)
from votekit.pref_profile import ProfileError, RankProfile
from votekit.pref_profile.profile_matrix import WeightKind


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("cleaned", [False, True])
def test_conversion_preserves_shared_rankings_voters_and_labels(exact, cleaned, monkeypatch):
    source = RankProfile(
        ballots=[
            RankBallot(ranking=[{"A", 1}, {"B"}], weight=2, voter_set={"voter"}),
            RankBallot(ranking=[{"B"}], weight=0),
        ],
        candidates=("B", "unused", 1, "A"),
        max_ranking_length=3,
    ).to_df()
    source.index = pd.Index(["same", "same"], name="Ballot Index")
    source["Weight"] = pd.Series(
        [2.0, 0.0] if exact else [Fraction(2, 3), Fraction(0)], index=source.index
    )
    profile = RankProfile(df=source, candidates=("B", "unused", 1, "A"), max_ranking_length=3)

    if cleaned:
        profile = remove_and_condense_rank_profile("A", profile, remove_zero_weight_ballots=False)

    expected_df = profile.to_df()
    converter = to_exact_fraction_weight if exact else float
    expected_df["Weight"] = expected_df["Weight"].map(converter)
    original_df = profile.to_df()
    original_total = profile.total_ballot_wt
    original_data = profile._matrix.voter_weight_data

    def reject_round_trip(*args, **kwargs):
        raise AssertionError(
            "Weight conversion must not export or use the public profile constructor"
        )

    with monkeypatch.context() as patch:
        patch.setattr(RankProfile, "to_df", reject_round_trip)
        patch.setattr(RankProfile, "__init__", reject_round_trip)

        converted = convert_profile_weights(profile, converter)

    assert type(converted) is RankProfile
    assert not hasattr(converted, "parent_profile")
    assert converted._matrix is not profile._matrix
    assert converted._matrix.rankings is profile._matrix.rankings
    assert converted._matrix.cand_sets is profile._matrix.cand_sets

    data = converted._matrix.voter_weight_data

    assert data.voter_sets is original_data.voter_sets
    assert data.ballot_index is original_data.ballot_index
    assert not np.shares_memory(data.weights, original_data.weights)
    assert not data.weights.flags.writeable
    assert data.weight_kind is (WeightKind.FRACTION if exact else WeightKind.FLOAT)
    assert converted.candidates == profile.candidates
    assert converted.max_ranking_length == profile.max_ranking_length
    assert converted.total_ballot_wt == sum(expected_df["Weight"])
    pd.testing.assert_frame_equal(converted.to_df(), expected_df)

    exported = converted.to_df()
    exported["Weight"] = 99.0

    pd.testing.assert_frame_equal(converted.to_df(), expected_df)
    pd.testing.assert_frame_equal(profile.to_df(), original_df)
    assert profile.total_ballot_wt == original_total


@pytest.mark.parametrize(
    "value", [-1, Fraction(-1, 2), float("nan"), float("inf"), True, "1", 2**53 + 1]
)
def test_conversion_rejects_invalid_callback_results(value):
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}])])

    with pytest.raises(ProfileError):
        convert_profile_weights(profile, lambda weight: value)

    assert profile.total_ballot_wt == 1


def test_conversion_rejects_mixed_fraction_and_nonintegral_float():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}], weight=1), RankBallot(ranking=[{"A"}], weight=2)]
    )

    with pytest.raises(ProfileError, match="Fraction"):
        convert_profile_weights(profile, lambda weight: Fraction(1, 2) if weight == 1 else 0.5)


def test_conversion_updates_active_candidates_and_total():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}], weight=2), RankBallot(ranking=[{"B"}], weight=0)],
        candidates=("A", "B", "unused"),
    )

    converted = convert_profile_weights(profile, lambda weight: 0 if weight else 3)

    assert converted.candidates_cast == ("B",)
    assert converted.candidates == profile.candidates
    assert converted.total_ballot_wt == 3
    assert profile.candidates_cast == ("A",)
    assert profile.total_ballot_wt == 2


@pytest.mark.parametrize("has_declared_candidates", [False, True])
def test_conversion_validates_newly_active_undeclared_candidates(has_declared_candidates):
    ballots = [RankBallot(ranking=[{"B"}], weight=0)]

    if has_declared_candidates:
        ballots.append(RankBallot(ranking=[{"A"}], weight=1))

    profile = RankProfile(ballots=ballots)

    if has_declared_candidates:
        with pytest.raises(ProfileError, match="not a subset"):
            convert_profile_weights(profile, lambda weight: 1)
    else:
        converted = convert_profile_weights(profile, lambda weight: 1)

        assert converted.candidates == ("B",)
        assert converted.candidates_cast == ("B",)

    assert profile.candidates == (("A",) if has_declared_candidates else ())


def test_conversion_passes_float_to_callback_and_propagates_its_error():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}])])
    error = RuntimeError("conversion failed")

    def fail(weight):
        assert isinstance(weight, float)

        raise error

    with pytest.raises(RuntimeError, match="conversion failed") as raised:
        convert_profile_weights(profile, fail)

    assert raised.value is error


def test_empty_conversion_does_not_call_converter():
    profile = RankProfile(candidates=("unused",), max_ranking_length=2)

    def fail(weight):
        raise AssertionError("Empty profile has no weights to convert")

    converted = convert_profile_weights(profile, fail)

    assert converted._matrix.rankings is profile._matrix.rankings
    assert converted.total_ballot_wt == 0
    assert converted.candidates == profile.candidates
    pd.testing.assert_frame_equal(converted.to_df(), profile.to_df())


@pytest.mark.parametrize("exact", [False, True])
def test_with_weights_owns_replacement_array(exact: bool) -> None:
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}], weight=1)])
    replacement = np.array([Fraction(1, 3)] if exact else [0.5], dtype=object if exact else float)
    converted = profile._with_weights(replacement)
    expected_weight = replacement[0]
    replacement[0] = 99

    assert converted.total_ballot_wt == expected_weight
    assert converted._matrix.voter_weight_data.weights[0] == expected_weight
    assert not converted._matrix.voter_weight_data.weights.flags.writeable
    assert profile.total_ballot_wt == 1
