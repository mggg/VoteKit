"""Tests for deprecated parameter backward compatibility across all election classes."""

import warnings

import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.elections import (
    SNTV,
    STV,
    Approval,
    BlockPlurality,
    BoostedRandomDictator,
    Borda,
    CondoBorda,
    Cumulative,
    FastSTV,
    GeneralRating,
    Limited,
    Plurality,
    PluralityVeto,
    RandomDictator,
    RankedPairs,
    Rating,
    Schulze,
    SequentialRCV,
    SerialVeto,
    SimultaneousVeto,
)
from votekit.pref_profile import RankProfile, ScoreProfile

# =====================
# == Shared fixtures ==
# =====================

rank_profile = RankProfile(
    ballots=[
        RankBallot(ranking=({"A"}, {"B"}, {"C"})),
        RankBallot(ranking=({"A"}, {"C"}, {"B"})),
        RankBallot(ranking=({"B"}, {"A"}, {"C"})),
    ],
    max_ranking_length=3,
)

score_profile = ScoreProfile(
    ballots=[
        ScoreBallot(scores={"A": 1, "B": 0, "C": 0}, weight=3),
        ScoreBallot(scores={"A": 0, "B": 1, "C": 0}, weight=2),
        ScoreBallot(scores={"A": 0, "B": 0, "C": 1}),
    ],
)

# Profile with budget=1 per ballot for Limited/Cumulative tests
limited_profile = ScoreProfile(
    ballots=[
        ScoreBallot(scores={"A": 1, "B": 0, "C": 0}, weight=3),
        ScoreBallot(scores={"A": 0, "B": 1, "C": 0}, weight=2),
        ScoreBallot(scores={"A": 0, "B": 0, "C": 1}),
    ],
)


# =============
# == Helpers ==
# =============


def _assert_deprecation_warning(cls, kwargs, expected_fragment="has been renamed"):
    """Instantiate *cls* with **kwargs and assert a DeprecationWarning is raised."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cls(**kwargs)
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1, f"No DeprecationWarning from {cls.__name__}"
    assert any(expected_fragment in str(x.message) for x in dep_warnings)


# =======================================
# == Ranking elections -- m -> n_seats ==
# =======================================


class TestPlurality:
    def test_old_m_works(self):
        _assert_deprecation_warning(Plurality, dict(profile=rank_profile, m=1))

    def test_old_m_value_used(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            e = Plurality(rank_profile, m=2)
        assert e.n_seats == 2

    def test_clash_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            Plurality(rank_profile, n_seats=1, m=2)

    def test_positional_clash_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            Plurality(rank_profile, 1, m=2)

    def test_default_without_either(self):
        e = Plurality(rank_profile)
        assert e.n_seats == 1


class TestSNTV:
    def test_old_m_works(self):
        _assert_deprecation_warning(SNTV, dict(profile=rank_profile, m=1))

    def test_clash_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            SNTV(rank_profile, n_seats=1, m=2)

    def test_default_without_either(self):
        e = SNTV(rank_profile)
        assert e.n_seats == 1


class TestBorda:
    def test_old_m_works(self):
        _assert_deprecation_warning(Borda, dict(profile=rank_profile, m=1))

    def test_clash_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            Borda(rank_profile, n_seats=1, m=2)

    def test_default_without_either(self):
        e = Borda(rank_profile)
        assert e.n_seats == 1


class TestCondoBorda:
    def test_old_m_works(self):
        _assert_deprecation_warning(CondoBorda, dict(profile=rank_profile, m=1))

    def test_clash_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            CondoBorda(rank_profile, n_seats=1, m=2)

    def test_default_without_either(self):
        e = CondoBorda(rank_profile)
        assert e.n_seats == 1


class TestRankedPairs:
    def test_old_m_works(self):
        _assert_deprecation_warning(RankedPairs, dict(profile=rank_profile, m=1))

    def test_clash_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            RankedPairs(rank_profile, n_seats=1, m=2)

    def test_default_without_either(self):
        e = RankedPairs(rank_profile)
        assert e.n_seats == 1


class TestSchulze:
    def test_old_m_works(self):
        _assert_deprecation_warning(Schulze, dict(profile=rank_profile, m=1))

    def test_clash_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            Schulze(rank_profile, n_seats=1, m=2)

    def test_default_without_either(self):
        e = Schulze(rank_profile)
        assert e.n_seats == 1


class TestRandomDictator:
    """n_seats is required here -- no default."""

    def test_old_m_works(self):
        _assert_deprecation_warning(RandomDictator, dict(profile=rank_profile, m=1))

    def test_clash_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            RandomDictator(rank_profile, n_seats=1, m=2)

    def test_missing_raises(self):
        with pytest.raises(TypeError, match="Missing required argument"):
            RandomDictator(rank_profile)


class TestBoostedRandomDictator:
    """n_seats is required here -- no default."""

    def test_old_m_works(self):
        _assert_deprecation_warning(BoostedRandomDictator, dict(profile=rank_profile, m=1))

    def test_clash_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            BoostedRandomDictator(rank_profile, n_seats=1, m=2)

    def test_missing_raises(self):
        with pytest.raises(TypeError, match="Missing required argument"):
            BoostedRandomDictator(rank_profile)


class TestPluralityVeto:
    def test_old_m_works(self):
        _assert_deprecation_warning(PluralityVeto, dict(profile=rank_profile, m=1))

    def test_clash_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            PluralityVeto(rank_profile, n_seats=1, m=2)

    def test_default_without_either(self):
        e = PluralityVeto(rank_profile)
        assert e.n_seats == 1


class TestSerialVeto:
    def test_old_m_works(self):
        _assert_deprecation_warning(SerialVeto, dict(profile=rank_profile, m=1))

    def test_clash_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            SerialVeto(rank_profile, n_seats=1, m=2)

    def test_default_without_either(self):
        e = SerialVeto(rank_profile)
        assert e.n_seats == 1


class TestSimultaneousVeto:
    def test_old_m_works(self):
        _assert_deprecation_warning(SimultaneousVeto, dict(profile=rank_profile, m=1))

    def test_clash_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            SimultaneousVeto(rank_profile, n_seats=1, m=2)

    def test_default_without_either(self):
        e = SimultaneousVeto(rank_profile)
        assert e.n_seats == 1


class TestSTV:
    def test_old_m_works(self):
        _assert_deprecation_warning(STV, dict(profile=rank_profile, m=1))

    def test_clash_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            STV(rank_profile, n_seats=1, m=2)

    def test_default_without_either(self):
        e = STV(rank_profile)
        assert e.n_seats == 1


class TestFastSTV:
    def test_old_m_works(self):
        _assert_deprecation_warning(FastSTV, dict(profile=rank_profile, m=1))

    def test_clash_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            FastSTV(rank_profile, n_seats=1, m=2)

    def test_default_without_either(self):
        e = FastSTV(rank_profile)
        assert e.n_seats == 1


class TestSequentialRCV:
    def test_old_m_works(self):
        _assert_deprecation_warning(SequentialRCV, dict(profile=rank_profile, m=1))

    def test_clash_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            SequentialRCV(rank_profile, n_seats=1, m=2)

    def test_default_without_either(self):
        e = SequentialRCV(rank_profile)
        assert e.n_seats == 1


class TestBlockPlurality:
    def test_old_m_works(self):
        _assert_deprecation_warning(BlockPlurality, dict(profile=rank_profile, m=1))

    def test_clash_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            BlockPlurality(rank_profile, n_seats=1, m=2)

    def test_default_without_either(self):
        e = BlockPlurality(rank_profile)
        assert e.n_seats == 1  # type: ignore[union-attr]


# =======================================
# == Approval election -- m -> n_seats ==
# =======================================


class TestApproval:
    def test_old_m_works(self):
        _assert_deprecation_warning(Approval, dict(profile=score_profile, m=1))

    def test_old_m_value_used(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            e = Approval(score_profile, m=2)
        assert e.n_seats == 2

    def test_clash_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            Approval(score_profile, n_seats=1, m=2)

    def test_default_without_either(self):
        e = Approval(score_profile)
        assert e.n_seats == 1


# ========================================================================
# == Score elections -- m -> n_seats, k -> per_candidate_limit / budget ==
# ========================================================================


class TestGeneralRating:
    def test_old_m_works(self):
        _assert_deprecation_warning(GeneralRating, dict(profile=score_profile, m=1))

    def test_old_k_works(self):
        _assert_deprecation_warning(GeneralRating, dict(profile=score_profile, k=1))

    def test_old_m_value_used(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            e = GeneralRating(score_profile, m=2)
        assert e.n_seats == 2

    def test_old_k_value_used(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            e = GeneralRating(score_profile, k=2)
        assert e.per_candidate_limit == 2

    def test_clash_m_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            GeneralRating(score_profile, n_seats=1, m=2)

    def test_clash_k_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            GeneralRating(score_profile, per_candidate_limit=1, k=2)

    def test_defaults_without_either(self):
        e = GeneralRating(score_profile)
        assert e.n_seats == 1
        assert e.per_candidate_limit == 1


class TestRating:
    def test_old_m_works(self):
        _assert_deprecation_warning(Rating, dict(profile=score_profile, per_candidate_limit=1, m=1))

    def test_old_k_works(self):
        _assert_deprecation_warning(Rating, dict(profile=score_profile, k=1))

    def test_clash_m_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            Rating(score_profile, n_seats=1, per_candidate_limit=1, m=2)

    def test_clash_k_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            Rating(score_profile, per_candidate_limit=1, k=2)


class TestLimited:
    def test_old_m_works(self):
        _assert_deprecation_warning(Limited, dict(profile=limited_profile, m=1))

    def test_old_k_works(self):
        """k maps to budget in Limited."""
        _assert_deprecation_warning(Limited, dict(profile=limited_profile, k=1))

    def test_old_k_value_used(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            e = Limited(limited_profile, n_seats=1, k=1)
        assert e.budget == 1

    def test_clash_m_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            Limited(limited_profile, n_seats=1, m=2)

    def test_clash_k_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            Limited(limited_profile, budget=1, k=1)

    def test_defaults_without_either(self):
        e = Limited(limited_profile)
        assert e.n_seats == 1
        assert e.budget == 1


class TestCumulative:
    def test_old_m_works(self):
        _assert_deprecation_warning(Cumulative, dict(profile=limited_profile, m=1))

    def test_clash_raises(self):
        with pytest.raises(TypeError, match="Cannot pass both"):
            Cumulative(limited_profile, n_seats=1, m=2)

    def test_default_without_either(self):
        e = Cumulative(limited_profile)
        assert e.n_seats == 1
