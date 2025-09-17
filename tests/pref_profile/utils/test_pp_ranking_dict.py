from votekit.ballot import RankBallot, ScoreBallot
from votekit.pref_profile import RankProfile, ScoreProfile
from votekit.pref_profile.utils import rank_profile_to_ranking_dict
import pytest


def test_to_ranking_dict():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"})),
            RankBallot(ranking=({"A"}, {"B"}), weight=3 / 2),
            RankBallot(ranking=({"C"}, {"B"}), weight=2),
        )
    )
    rv = rank_profile_to_ranking_dict(profile, standardize=False)
    assert rv[(frozenset({"A"}), frozenset({"B"}))] == 5 / 2
    assert rv[(frozenset({"C"}), frozenset({"B"}))] == 2 / 1

    rv = rank_profile_to_ranking_dict(profile, standardize=True)
    assert rv[(frozenset({"A"}), frozenset({"B"}))] == 5 / 9
    assert rv[(frozenset({"C"}), frozenset({"B"}))] == 4 / 9


def test_ranking_dict_warn():
    profile = ScoreProfile(ballots=(ScoreBallot(scores={"A": 4}),))

    with pytest.raises(
        TypeError,
        match=("Profile must be a RankProfile"),
    ):
        rank_profile_to_ranking_dict(profile)
