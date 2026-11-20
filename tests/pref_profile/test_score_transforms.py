import numpy as np
import pandas as pd
import pytest

from votekit.ballot import ScoreBallot
from votekit.pref_profile import ScoreProfile
from votekit.pref_profile.profile_transform import JitterScores


def score_weights(profile: ScoreProfile) -> dict[tuple[float | None, ...], float]:
    """Total weight of each (A, B) score pair, with unscored (NaN) shown as None."""
    weights: dict[tuple[float | None, ...], float] = {}
    for _, row in profile.to_df().iterrows():
        scores = tuple(None if np.isnan(row[c]) else float(row[c]) for c in ("A", "B"))
        weights[scores] = weights.get(scores, 0.0) + float(row["Weight"])
    return weights


def test_jitter_moves_exactly_one_score_by_one_step():
    profile = ScoreProfile(
        ballots=[ScoreBallot(scores={"A": 2, "B": 1}, weight=200)], candidates=["A", "B"]
    )

    perturbed = profile.perturb(JitterScores(rng_seed=0))

    assert set(score_weights(perturbed)) == {(3.0, 1.0), (1.0, 1.0), (2.0, 2.0), (2.0, 0.0)}


def test_jitter_draws_each_move_with_equal_probability():
    profile = ScoreProfile(
        ballots=[ScoreBallot(scores={"A": 2, "B": 1}, weight=1200)], candidates=["A", "B"]
    )

    perturbed = profile.perturb(JitterScores(rng_seed=0))

    weights = score_weights(perturbed)
    for scores in [(3.0, 1.0), (1.0, 1.0), (2.0, 2.0), (2.0, 0.0)]:
        assert weights[scores] / 1200 == pytest.approx(1 / 4, abs=0.05)


def test_jitter_keeps_an_unscored_candidate_unscored_and_stops_at_zero():
    # A is unscored and B scores 0, so A never changes and B can only move up
    profile = ScoreProfile(
        ballots=[ScoreBallot(scores={"B": 0}, weight=200)], candidates=["A", "B"]
    )

    perturbed = profile.perturb(JitterScores(rng_seed=0))

    assert set(score_weights(perturbed)) == {(None, 0.0), (None, 1.0)}


def test_jitter_stops_at_max_score():
    profile = ScoreProfile(
        ballots=[ScoreBallot(scores={"A": 3, "B": 3}, weight=200)], candidates=["A", "B"]
    )

    perturbed = profile.perturb(JitterScores(rng_seed=0, max_score=3.0))

    assert set(score_weights(perturbed)) == {(3.0, 3.0), (2.0, 3.0), (3.0, 2.0)}


def test_jitter_never_changes_an_undeclared_column():
    # Z is a score column but not a declared candidate
    df = pd.DataFrame(
        {"A": [1.0], "B": [1.0], "Z": [np.nan], "Voter Set": [set()], "Weight": [100.0]}
    )
    df.index.name = "Ballot Index"
    profile = ScoreProfile(df=df, candidates=["A", "B"])

    perturbed = profile.perturb(JitterScores(rng_seed=0))

    assert perturbed.to_df()["Z"].isna().all()


def test_jitter_rejects_a_score_already_above_max_score():
    profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": 4, "B": 1})], candidates=["A", "B"])

    with pytest.raises(ValueError, match="max_score 3.0 is below max score given in the matrix"):
        profile.perturb(JitterScores(rng_seed=0, max_score=3.0))


def test_jitter_rejects_a_non_positive_step():
    with pytest.raises(ValueError, match="step must be a positive finite number"):
        JitterScores(step=0)


def test_jitter_rejects_a_non_finite_step():
    with pytest.raises(ValueError, match="step must be a positive finite number"):
        JitterScores(step=float("nan"))
    with pytest.raises(ValueError, match="step must be a positive finite number"):
        JitterScores(step=float("inf"))


def test_jitter_rejects_a_negative_max_score():
    with pytest.raises(ValueError, match="max_score must be a non-negative finite number"):
        JitterScores(max_score=-1)


def test_jitter_rejects_a_non_finite_max_score():
    # a NaN cap would otherwise turn every jittered score into NaN
    with pytest.raises(ValueError, match="max_score must be a non-negative finite number"):
        JitterScores(max_score=float("nan"))
    with pytest.raises(ValueError, match="max_score must be a non-negative finite number"):
        JitterScores(max_score=float("inf"))


def test_jitter_is_reproducible_and_a_reused_transform_continues_its_stream():
    profile = ScoreProfile(
        ballots=[ScoreBallot(scores={"A": 2, "B": 1}, weight=50)], candidates=["A", "B"]
    )

    first = score_weights(profile.perturb(JitterScores(rng_seed=1)))
    again = score_weights(profile.perturb(JitterScores(rng_seed=1)))
    assert first == again

    reused = JitterScores(rng_seed=1)
    assert score_weights(profile.perturb(reused)) != score_weights(profile.perturb(reused))
