from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.elections import STV, Borda, GeneralRating
from votekit.pref_profile import ProfileError, RankProfile, ScoreProfile
from votekit.pref_profile.profile_matrix import RankMatrix, ScoreMatrix


def _reject_materialization(*args: object, **kwargs: object) -> None:
    raise AssertionError("Validation must read matrix storage directly")


@pytest.fixture
def forbid_materialization(monkeypatch: pytest.MonkeyPatch) -> None:
    for profile_type in (RankProfile, ScoreProfile):
        monkeypatch.setattr(profile_type, "to_df", _reject_materialization)
        monkeypatch.setattr(profile_type, "ballots", property(_reject_materialization))
    for matrix_type in (RankMatrix, ScoreMatrix):
        monkeypatch.setattr(matrix_type, "to_df", _reject_materialization)


def test_rank_validation_reads_ids(forbid_materialization: None) -> None:
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {1}])])
    object.__new__(Borda)._validate_profile(profile)
    object.__new__(STV)._stv_validate_profile(profile)


@pytest.mark.parametrize("empty", [False, True])
def test_rank_validation_rejects_empty_and_unranked_profiles(
    empty: bool,
    forbid_materialization: None,
) -> None:
    profile = RankProfile() if empty else RankProfile(ballots=[RankBallot()])
    with pytest.raises(ProfileError, match="at least one"):
        object.__new__(Borda)._validate_profile(profile)


def test_rank_validation_reports_unranked_ballot_without_materializing_all_ballots(
    forbid_materialization: None,
) -> None:
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}]), RankBallot(voter_set={"v"})])
    with pytest.raises(TypeError, match="has no ranking"):
        object.__new__(Borda)._validate_profile(profile)
    with pytest.raises(TypeError, match="Ballots must have rankings"):
        object.__new__(STV)._stv_validate_profile(profile)


def test_stv_tie_diagnostic_uses_row_position_with_duplicate_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}], weight=1),
            RankBallot(ranking=[{"A", "B"}], weight=7),
        ],
        max_ranking_length=2,
    ).to_df()
    source.index = pd.Index([42, 42], name="Ballot Index")
    profile = RankProfile(df=source, candidates=("A", "B"), max_ranking_length=2)
    monkeypatch.setattr(RankProfile, "to_df", _reject_materialization)
    monkeypatch.setattr(RankProfile, "ballots", property(_reject_materialization))
    with pytest.raises(TypeError, match="contains a tied ranking") as raised:
        object.__new__(STV)._stv_validate_profile(profile)
    assert "7" in str(raised.value)


def _rating_validator(limit: float = 2, budget: float | None = None) -> GeneralRating:
    validator = object.__new__(GeneralRating)
    validator.per_candidate_limit = limit
    validator.budget = budget
    return validator


def test_rating_validation_reads_scores(forbid_materialization: None) -> None:
    profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": 2, 1: 1})])
    _rating_validator(budget=3)._validate_profile(profile)


@pytest.mark.parametrize(
    "scores,message",
    [
        (None, "score dictionary"),
        ({"A": 3}, "score limit"),
        ({"A": -1}, "non-negative"),
        ({"A": 2, "B": 2}, "total score budget"),
    ],
)
def test_rating_validation_reports_invalid_scores(
    scores: dict[str, float] | None,
    message: str,
    forbid_materialization: None,
) -> None:
    profile = ScoreProfile(ballots=[ScoreBallot(scores=scores)])
    with pytest.raises(TypeError, match=message):
        _rating_validator(budget=3)._validate_profile(profile)


def test_rating_validation_preserves_first_invalid_row_and_check_order(
    forbid_materialization: None,
) -> None:
    profile = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 3, "B": -1}),
            ScoreBallot(),
        ]
    )
    with pytest.raises(TypeError, match="score limit"):
        _rating_validator(budget=3)._validate_profile(profile)


def test_rating_validation_distinguishes_zero_from_missing_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = pd.DataFrame({"A": [0.0], "B": [np.nan], "Weight": [1.0], "Voter Set": [frozenset()]})
    source.index.name = "Ballot Index"
    profile = ScoreProfile(df=source, candidates=("A", "B"))
    monkeypatch.setattr(ScoreProfile, "to_df", _reject_materialization)
    monkeypatch.setattr(ScoreProfile, "ballots", property(_reject_materialization))
    _rating_validator()._validate_profile(profile)


def test_rating_budget_uses_python_float_sum(forbid_materialization: None) -> None:
    scores = {"A": 0.1, "B": 0.2, "C": 0.3}
    profile = ScoreProfile(ballots=[ScoreBallot(scores=scores)])
    validator = _rating_validator(limit=0.3, budget=0.6)
    if sum(scores.values()) > 0.6:
        with pytest.raises(TypeError, match="total score budget"):
            validator._validate_profile(profile)
    else:
        validator._validate_profile(profile)
