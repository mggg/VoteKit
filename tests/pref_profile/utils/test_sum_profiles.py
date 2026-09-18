from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import pandas as pd
import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.cleaning import clean_rank_profile
from votekit.exceptions import ProfileError
from votekit.pref_profile import RankProfile, ScoreProfile
from votekit.pref_profile.profile_matrix import RankMatrix, ScoreMatrix
from votekit.pref_profile.utils import sum_profiles


def test_sum_profiles_with_mixed_types_raises_type_error():
    score_profile = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2, "B": 2}, weight=2),
            ScoreBallot(scores={"A": 2, "C": 2}, voter_set={"Chris"}),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
    )
    rank_profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
            RankBallot(),
            RankBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
        max_ranking_length=3,
    )
    with pytest.raises(
        TypeError,
        match="All profiles must be of the same type.",
    ):
        sum_profiles([score_profile, score_profile, score_profile, rank_profile])

    with pytest.raises(
        TypeError,
        match="All profiles must be of the same type.",
    ):
        sum_profiles([rank_profile, score_profile, score_profile, rank_profile])


def test_sum_empty_profile_raises_value_error():
    with pytest.raises(
        ValueError,
        match="Cannot sum an empty list of profiles",
    ):
        sum_profiles([])


def test_sum_one_profile_returns_copy_of_same_profile():
    profile = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2, "B": 2}, weight=2),
            ScoreBallot(scores={"A": 2, "C": 2}, voter_set={"Chris"}),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
    )
    summed_profile = sum_profiles([profile])
    assert summed_profile == profile
    assert id(summed_profile) != id(profile)

    profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
            RankBallot(),
            RankBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
        max_ranking_length=3,
    )
    summed_profile = sum_profiles([profile])
    assert summed_profile == profile
    assert id(summed_profile) != id(profile)


def test_sum_one_profile_no_list_raises_type_error():
    profile = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2, "B": 2}, weight=2),
            ScoreBallot(scores={"A": 2, "C": 2}, voter_set={"Chris"}),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
    )
    with pytest.raises(TypeError, match="has no len()"):
        sum_profiles(profile)  # type: ignore[arg-type]

    profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
            RankBallot(),
            RankBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
        max_ranking_length=3,
    )
    with pytest.raises(TypeError, match="has no len()"):
        sum_profiles(profile)  # type: ignore[arg-type]


def test_sum_score_profiles():
    profile_1 = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2, "B": 2}, weight=2),
            ScoreBallot(scores={"A": 2, "C": 2}, voter_set={"Chris"}),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
    )

    profile_2 = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"D": 2, "E": 2}, weight=2),
            ScoreBallot(scores={"D": 2, "E": 2, "F": 3.1}, weight=2),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ],
        candidates=["D", "E", "F"],
    )

    profile_3 = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"G": 2, "H": 2}, weight=2),
            ScoreBallot(scores={"G": 2, "H": 2, "I": 3.1}, weight=2),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ],
        candidates=["G", "H", "I"],
    )
    summed_profile = sum_profiles([profile_1, profile_2, profile_3])
    true_summed_profile = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2, "B": 2}, weight=2),
            ScoreBallot(scores={"A": 2, "C": 2}, voter_set={"Chris"}),
            ScoreBallot(),
            ScoreBallot(weight=0),
            ScoreBallot(scores={"D": 2, "E": 2}, weight=2),
            ScoreBallot(scores={"D": 2, "E": 2, "F": 3.1}, weight=2),
            ScoreBallot(),
            ScoreBallot(weight=0),
            ScoreBallot(scores={"G": 2, "H": 2}, weight=2),
            ScoreBallot(scores={"G": 2, "H": 2, "I": 3.1}, weight=2),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
    )

    assert set(summed_profile.candidates) == set(["A", "B", "C", "D", "E", "F", "G", "H", "I"])
    assert isinstance(summed_profile, ScoreProfile)
    assert true_summed_profile == summed_profile


def test_sum_rank_profiles():
    profile_1 = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
            RankBallot(),
            RankBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
        max_ranking_length=3,
    )

    profile_2 = RankProfile(
        ballots=[
            RankBallot(ranking=({"E"}, {"D"}, {"F"}, {"E"}), weight=2),
            RankBallot(ranking=({"D"}, {"E"}, {"F"}), weight=2),
            RankBallot(),
            RankBallot(weight=0),
        ],
        candidates=["D", "E", "F"],
    )

    profile_3 = RankProfile(
        ballots=[
            RankBallot(ranking=({"G"}, {"H"}, {"I"}, {"G"}), weight=2),
            RankBallot(ranking=({"G"}, {"H"}, {"I"}), weight=2),
            RankBallot(),
            RankBallot(weight=0),
        ],
        candidates=["G", "H", "I"],
    )
    summed_profile = sum_profiles([profile_1, profile_2, profile_3])
    true_summed_profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
            RankBallot(),
            RankBallot(weight=0),
            RankBallot(ranking=({"E"}, {"D"}, {"F"}, {"E"}), weight=2),
            RankBallot(ranking=({"D"}, {"E"}, {"F"}), weight=2),
            RankBallot(),
            RankBallot(weight=0),
            RankBallot(ranking=({"G"}, {"H"}, {"I"}, {"G"}), weight=2),
            RankBallot(ranking=({"G"}, {"H"}, {"I"}), weight=2),
            RankBallot(),
            RankBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        max_ranking_length=4,
    )

    assert set(summed_profile.candidates) == set(["A", "B", "C", "D", "E", "F", "G", "H", "I"])
    assert summed_profile.max_ranking_length == 4
    assert isinstance(summed_profile, RankProfile)
    assert true_summed_profile == summed_profile


def _reject_public_profile_construction(*args: object, **kwargs: object) -> None:
    raise AssertionError("Profile addition must not call the public profile constructor")


def test_sum_rank_profiles_aligns_payloads_and_weights_without_dataframe_conversion(
    monkeypatch: pytest.MonkeyPatch, unexpected_dataframe_conversion: Callable[..., None]
) -> None:
    ballots = [
        RankBallot(ranking=[{"B"}], weight=Fraction(1, 3), voter_set={"Alice"}),
        RankBallot(ranking=[{"A"}, {"B"}], weight=2.0, voter_set={"Bob"}),
    ]
    profiles = [
        RankProfile(ballots=[ballots[0]], candidates=("B",)),
        RankProfile(ballots=[ballots[1]], candidates=("A", "B", "unused")),
    ]
    sources = [profile.to_df() for profile in profiles]

    with monkeypatch.context() as patch:
        for cls in (RankProfile, ScoreProfile, RankMatrix, ScoreMatrix):
            patch.setattr(cls, "to_df", unexpected_dataframe_conversion)

        patch.setattr(RankProfile, "__init__", _reject_public_profile_construction)

        combined = sum_profiles(profiles)

    assert combined.ballots == tuple(ballots)
    assert all(isinstance(ballot.weight, Fraction) for ballot in combined.ballots)
    assert combined.total_ballot_wt == Fraction(7, 3)
    assert set(combined.candidates) == {"A", "B", "unused"}
    assert combined.to_df().index.tolist() == [0, 1]

    for profile, expected in zip(profiles, sources, strict=True):
        pd.testing.assert_frame_equal(profile.to_df(), expected)

    assert isinstance(combined, RankProfile)
    assert combined.max_ranking_length == 2
    assert len(set(combined._matrix.cand_sets)) == len(combined._matrix.cand_sets)


def test_sum_rank_profiles_rejects_mixed_fraction_nonintegral_float(
    forbid_dataframe_conversion: Callable[[], None],
) -> None:
    profiles = [
        RankProfile(ballots=[RankBallot(ranking=[{"A"}], weight=weight)])
        for weight in (Fraction(1, 3), 0.5)
    ]

    forbid_dataframe_conversion()

    with pytest.raises(ProfileError, match="Cannot mix Fraction"):
        sum_profiles(profiles)


def test_sum_rank_profiles_keeps_empty_profiles_and_missing_voters_aligned(
    monkeypatch: pytest.MonkeyPatch, unexpected_dataframe_conversion: Callable[..., None]
) -> None:
    empty = RankProfile(candidates=("unused",))
    source = RankProfile(ballots=[RankBallot(ranking=[{"A"}])]).to_df()
    source.index = pd.Index([42], name="Ballot Index")
    without_voters = RankProfile(df=source, candidates=("A",), max_ranking_length=1)

    source["Voter Set"] = pd.Series([frozenset({"v"})], index=source.index)
    with_voters = RankProfile(df=source, candidates=("A",), max_ranking_length=1)

    with monkeypatch.context() as patch:
        for cls in (RankProfile, ScoreProfile, RankMatrix, ScoreMatrix):
            patch.setattr(cls, "to_df", unexpected_dataframe_conversion)

        combined = sum_profiles([empty, without_voters, empty, with_voters])
        all_empty = sum_profiles([empty, empty])

    assert combined.to_df().index.tolist() == [0, 1]
    assert [ballot.voter_set for ballot in combined.ballots] == [frozenset(), frozenset({"v"})]
    assert combined.ballots[0] == RankBallot(ranking=[{"A"}])
    assert set(combined.candidates) == {"A", "unused"}

    assert all_empty.ballots == ()
    assert all_empty.candidates == ("unused",)


def test_sum_score_profiles_aligns_payloads_and_weights_without_dataframe_conversion(
    monkeypatch: pytest.MonkeyPatch, unexpected_dataframe_conversion: Callable[..., None]
) -> None:
    ballots = [
        ScoreBallot(scores={"B": 2}, weight=Fraction(1, 3), voter_set={"Alice"}),
        ScoreBallot(scores={"A": 3, "B": 1}, weight=2.0, voter_set={"Bob"}),
    ]
    profiles = [
        ScoreProfile(ballots=[ballots[0]], candidates=("B",)),
        ScoreProfile(ballots=[ballots[1]], candidates=("A", "B", "unused")),
    ]
    sources = [profile.to_df() for profile in profiles]

    with monkeypatch.context() as patch:
        for cls in (RankProfile, ScoreProfile, RankMatrix, ScoreMatrix):
            patch.setattr(cls, "to_df", unexpected_dataframe_conversion)

        patch.setattr(ScoreProfile, "__init__", _reject_public_profile_construction)

        combined = sum_profiles(profiles)

    assert combined.ballots == tuple(ballots)
    assert all(isinstance(ballot.weight, Fraction) for ballot in combined.ballots)
    assert combined.total_ballot_wt == Fraction(7, 3)
    assert set(combined.candidates) == {"A", "B", "unused"}
    assert combined.to_df().index.tolist() == [0, 1]

    for profile, expected in zip(profiles, sources, strict=True):
        pd.testing.assert_frame_equal(profile.to_df(), expected)


def test_sum_score_profiles_rejects_mixed_fraction_nonintegral_float(
    forbid_dataframe_conversion: Callable[[], None],
) -> None:
    profiles = [
        ScoreProfile(ballots=[ScoreBallot(scores={"A": 1}, weight=weight)])
        for weight in (Fraction(1, 3), 0.5)
    ]

    forbid_dataframe_conversion()

    with pytest.raises(ProfileError, match="Cannot mix Fraction"):
        sum_profiles(profiles)


def test_sum_score_profiles_keeps_empty_profiles_and_missing_voters_aligned(
    monkeypatch: pytest.MonkeyPatch, unexpected_dataframe_conversion: Callable[..., None]
) -> None:
    empty = ScoreProfile(candidates=("unused",))
    source = ScoreProfile(ballots=[ScoreBallot(scores={"A": 2})]).to_df()
    source.index = pd.Index([42], name="Ballot Index")
    without_voters = ScoreProfile(df=source, candidates=("A",), max_ranking_length=1)

    source["Voter Set"] = pd.Series([frozenset({"v"})], index=source.index)
    with_voters = ScoreProfile(df=source, candidates=("A",), max_ranking_length=1)

    with monkeypatch.context() as patch:
        for cls in (RankProfile, ScoreProfile, RankMatrix, ScoreMatrix):
            patch.setattr(cls, "to_df", unexpected_dataframe_conversion)

        combined = sum_profiles([empty, without_voters, empty, with_voters])
        all_empty = sum_profiles([empty, empty])

    assert combined.to_df().index.tolist() == [0, 1]
    assert [ballot.voter_set for ballot in combined.ballots] == [frozenset(), frozenset({"v"})]
    assert combined.ballots[0] == ScoreBallot(scores={"A": 2})
    assert set(combined.candidates) == {"A", "unused"}

    assert all_empty.ballots == ()
    assert all_empty.candidates == ("unused",)


def test_score_sum_aligns_columns_and_uses_each_source_candidate_universe(
    forbid_dataframe_conversion: Callable[[], None],
) -> None:
    source = pd.DataFrame(
        {
            "B": [9.0, 0.0],
            "A": [2.0, 3.0],
            "Weight": [0.0, 1.0],
            "Voter Set": [frozenset(), frozenset()],
        }
    )
    source.index.name = "Ballot Index"
    first = ScoreProfile(df=source, candidates=("A",))
    second = ScoreProfile(ballots=[ScoreBallot(scores={"B": 4, "A": 1})], candidates=("B", "A"))

    forbid_dataframe_conversion()

    combined = sum_profiles([first, second])

    assert combined.ballots == (
        ScoreBallot(scores={"A": 2}, weight=0),
        ScoreBallot(scores={"A": 3}),
        ScoreBallot(scores={"A": 1, "B": 4}),
    )


def test_sum_rank_profile_with_reduced_max_ranking_length():
    original_profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
            RankBallot(),
            RankBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
        max_ranking_length=5,
    )
    reduced_profile = clean_rank_profile(
        original_profile,
        lambda ranking: ranking,
        remove_null_ballots=False,
        remove_zero_weight_ballots=False,
        reduce_max_ranking_length=True,
    )

    assert reduced_profile.max_ranking_length == 3
    assert [col for col in reduced_profile.to_df().columns if col.startswith("Ranking_")] == [
        "Ranking_1",
        "Ranking_2",
        "Ranking_3",
    ]

    profile_2 = RankProfile(
        ballots=[
            RankBallot(ranking=({"E"}, {"D"}, {"F"}, {"E"}), weight=2),
            RankBallot(ranking=({"D"}, {"E"}, {"F"}), weight=2),
            RankBallot(),
            RankBallot(weight=0),
        ],
        candidates=["D", "E", "F"],
        max_ranking_length=4,
    )

    true_summed_profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
            RankBallot(),
            RankBallot(weight=0),
            RankBallot(ranking=({"E"}, {"D"}, {"F"}, {"E"}), weight=2),
            RankBallot(ranking=({"D"}, {"E"}, {"F"}), weight=2),
            RankBallot(),
            RankBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D", "E", "F"],
        max_ranking_length=4,
    )

    summed_profile = reduced_profile + profile_2
    assert summed_profile.max_ranking_length == 4
    assert summed_profile == true_summed_profile
