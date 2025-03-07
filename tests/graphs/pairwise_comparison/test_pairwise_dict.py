from votekit.graphs import pairwise_dict, restrict_pairwise_dict_to_subset
from votekit.pref_profile import PreferenceProfile
from votekit.ballot import Ballot
import pytest

ballots = [
    Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=10),
    Ballot(ranking=[{"A"}, {"C"}, {"B"}], weight=10),
    Ballot(ranking=[{"B"}, {"A"}, {"C"}], weight=10),
]
test_profile = PreferenceProfile(ballots=ballots)


def test_pairwise_dict():
    pwd = pairwise_dict(test_profile)

    assert pwd[("A", "B")] == [10, 20]
    assert pwd[("A", "C")] == [20, 10]
    assert pwd[("B", "C")] == [10, 20]


def test_restrict_pairwise():
    pwd = pairwise_dict(test_profile)
    restricted_pwd = restrict_pairwise_dict_to_subset(["A", "B"], pwd)

    assert restricted_pwd[("A", "B")] == [10, 20]
    assert ("A", "C") not in restricted_pwd
    assert ("B", "C") not in restricted_pwd


def test_restrict_pairwise_single_cand():
    # make sure passing as string doesn't mess this up
    ballots = [
        Ballot(ranking=[{"Chris"}, {"Peter"}, {"Moon"}], weight=10),
    ]
    profile = PreferenceProfile(ballots=ballots)

    pwd = pairwise_dict(profile)

    with pytest.raises(
        ValueError, match="Must be at least two candidates in cand_subset:"
    ):
        restrict_pairwise_dict_to_subset(["Chris"], pwd)


def test_restrict_pairwise_cand_error():
    pwd = pairwise_dict(test_profile)
    with pytest.raises(
        ValueError,
        match=(
            "are found in cand_subset but "
            "not in the list of candidates found in the dictionary:"
        ),
    ):
        restrict_pairwise_dict_to_subset(["A", "E"], pwd)
