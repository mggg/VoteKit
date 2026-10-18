from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.exceptions import ProfileError
from votekit.pref_profile import RankProfile, ScoreProfile
from votekit.pref_profile.profile_matrix import ProfileVoterWeightData, WeightKind


def _profile_from_weights(profile_type, weights: pd.Series):
    df = pd.DataFrame({"Weight": weights, "Voter Set": [frozenset()] * len(weights)})
    if profile_type is RankProfile:
        df["Ranking_1"] = pd.Series(
            [frozenset({"A" if row % 2 == 0 else "B"}) for row in range(len(weights))],
            dtype=object,
        )
    else:
        df["A"] = [int(row % 2 == 0) for row in range(len(weights))]
        df["B"] = [int(row % 2 == 1) for row in range(len(weights))]
    df.index.name = "Ballot Index"
    return profile_type(df=df, candidates=["A", "B"], max_ranking_length=1)


@pytest.mark.parametrize("profile_type", [RankProfile, ScoreProfile])
@pytest.mark.parametrize("weight", [0.1, 1e-9, np.float64(1.5)])
def test_profile_rejects_fraction_mixed_with_non_integral_float(profile_type, weight):
    with pytest.raises(ProfileError, match="Cannot mix Fraction weights with non-integral"):
        _profile_from_weights(profile_type, pd.Series([weight, Fraction(1)], dtype=object))


@pytest.mark.parametrize("profile_type", [RankProfile, ScoreProfile])
@pytest.mark.parametrize("weight", [0, 2.0, np.float64(2), 2**53 + 1, 10**400])
def test_profile_converts_integral_weights_to_fractions_exactly(profile_type, weight):
    expected = [Fraction(int(weight)), Fraction(1, 3)]
    profile = _profile_from_weights(profile_type, pd.Series([weight, expected[1]], dtype=object))

    assert profile._matrix.voter_weight_data.weight_kind is WeightKind.FRACTION
    assert profile.to_df()["Weight"].tolist() == expected
    assert [ballot.weight for ballot in profile.ballots] == expected
    assert all(isinstance(ballot.weight, Fraction) for ballot in profile.ballots)
    assert profile.total_ballot_wt == sum(expected)
    assert set(profile.candidates_cast) == ({"A", "B"} if weight > 0 else {"B"})


@pytest.mark.parametrize("profile_type", [RankProfile, ScoreProfile])
@pytest.mark.parametrize("dtype", [None, object])
@pytest.mark.parametrize("weight", [np.int64(2**53 + 1), np.uint64(2**64 - 1)])
def test_profile_rejects_integer_rounding_in_numeric_and_object_columns(
    profile_type, dtype, weight
):
    with pytest.raises(ProfileError, match="represented exactly as float64"):
        _profile_from_weights(profile_type, pd.Series([weight, weight], dtype=dtype))


@pytest.mark.parametrize("profile_type", [RankProfile, ScoreProfile])
def test_profile_rejects_integer_overflow_without_fraction_storage(profile_type):
    with pytest.raises(ProfileError, match="represented exactly as float64"):
        _profile_from_weights(profile_type, pd.Series([10**400, 1], dtype=object))


@pytest.mark.parametrize("profile_type", [RankProfile, ScoreProfile])
@pytest.mark.parametrize("weight_type", [np.float16, np.float32, np.longdouble])
@pytest.mark.parametrize("object_column", [False, True])
def test_profile_rejects_unsupported_floating_weight_types(
    profile_type, weight_type, object_column
):
    dtype = object if object_column else weight_type
    weights = pd.Series([weight_type(1.5), weight_type(2)], dtype=dtype)

    with pytest.raises(ProfileError, match="float64"):
        _profile_from_weights(profile_type, weights)


@pytest.mark.parametrize("profile_type", [RankProfile, ScoreProfile])
@pytest.mark.parametrize("weight", [0.1, np.float64(1.5), np.uint64(2**63), 1e-9])
@pytest.mark.parametrize("dtype", [None, object])
def test_profile_accepts_exact_float64_conversion(profile_type, weight, dtype):
    profile = _profile_from_weights(profile_type, pd.Series([weight, weight], dtype=dtype))

    assert profile._matrix.voter_weight_data.weight_kind is WeightKind.FLOAT
    assert profile.to_df()["Weight"].tolist() == [float(weight), float(weight)]
    assert profile.total_ballot_wt == 2 * float(weight)
    assert set(profile.candidates_cast) == {"A", "B"}


@pytest.mark.parametrize("dtype", [None, object])
@pytest.mark.parametrize("weight", [True, np.bool_(True), 1 + 2j])
def test_weight_adapter_rejects_boolean_and_complex_values(dtype, weight):
    df = pd.DataFrame({"Weight": pd.Series([weight], dtype=dtype), "Voter Set": [frozenset()]})

    with pytest.raises(ProfileError, match="real numbers|float, int, or Fraction"):
        ProfileVoterWeightData.from_df(df)


@pytest.mark.parametrize(
    "profile_type, ballot_type, payload",
    [
        (RankProfile, RankBallot, {"ranking": [{"A"}]}),
        (ScoreProfile, ScoreBallot, {"scores": {"A": 1}}),
    ],
)
def test_ballot_input_obeys_mixed_weight_policy(profile_type, ballot_type, payload):
    rational = ballot_type(**payload, weight=Fraction(1, 3))
    fractional_float = ballot_type(**payload, weight=0.1)
    with pytest.raises(ProfileError, match="Cannot mix Fraction weights with non-integral"):
        profile_type(ballots=[rational, fractional_float])

    profile = profile_type(ballots=[rational, ballot_type(**payload, weight=2)])
    assert profile.total_ballot_wt == Fraction(7, 3)
    assert [ballot.weight for ballot in profile.ballots] == [Fraction(1, 3), Fraction(2)]


@pytest.mark.parametrize("profile_type", [RankProfile, ScoreProfile])
@pytest.mark.parametrize("dtype", [float, object])
def test_empty_profile_weights_use_float_storage(profile_type, dtype):
    profile = _profile_from_weights(profile_type, pd.Series([], dtype=dtype))

    assert profile._matrix.voter_weight_data.weight_kind is WeightKind.FLOAT
    assert profile.total_ballot_wt == 0
    assert profile.candidates_cast == ()
