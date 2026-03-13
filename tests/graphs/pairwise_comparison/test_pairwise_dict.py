from typing import cast

import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.graphs import pairwise_dict, restrict_pairwise_dict_to_subset
from votekit.pref_profile import RankProfile, ScoreProfile

ballots = (
    RankBallot(ranking=tuple(map(frozenset, [{"C"}, {"B"}, {"A"}])), weight=10),
    RankBallot(ranking=tuple(map(frozenset, [{"A"}, {"C"}, {"B"}])), weight=10),
    RankBallot(ranking=tuple(map(frozenset, [{"B"}, {"A"}, {"C"}])), weight=10),
)
test_profile = RankProfile(ballots=ballots)


def test_pairwise_dict():
    pwd = pairwise_dict(test_profile, sort_candidate_pairs=True)

    assert pwd[("A", "B")] == (10, 20)
    assert pwd[("A", "C")] == (20, 10)
    assert pwd[("B", "C")] == (10, 20)


def test_restrict_pairwise():
    pwd = pairwise_dict(test_profile, sort_candidate_pairs=True)
    restricted_pwd = restrict_pairwise_dict_to_subset(["A", "B"], pwd)

    assert restricted_pwd[("A", "B")] == (10, 20)
    assert ("A", "C") not in restricted_pwd
    assert ("B", "C") not in restricted_pwd


def test_restrict_pairwise_single_cand():
    # make sure passing as string doesn't mess this up
    ballots = [
        RankBallot(ranking=tuple(map(frozenset, [{"Chris"}, {"Peter"}, {"Moon"}])), weight=10),
    ]
    profile = RankProfile(ballots=tuple(ballots))

    pwd = pairwise_dict(profile)

    with pytest.raises(ValueError, match="Must be at least two candidates in cand_subset:"):
        restrict_pairwise_dict_to_subset(["Chris"], pwd)


def test_restrict_pairwise_cand_error():
    pwd = pairwise_dict(test_profile)
    with pytest.raises(
        ValueError,
        match=(
            "are found in cand_subset but " "not in the list of candidates found in the dictionary:"
        ),
    ):
        restrict_pairwise_dict_to_subset(["A", "E"], pwd)


def test_pairwise_contains_rankings_errors():
    with pytest.raises(ValueError, match="Profile must be of type RankProfile."):
        pairwise_dict(cast(RankProfile, ScoreProfile(ballots=(ScoreBallot(scores={"Chris": 4}),))))

    with pytest.raises(ValueError, match="All ballots must have rankings."):
        pairwise_dict(RankProfile(ballots=[RankBallot()]))
