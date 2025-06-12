from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile, ProfileError
import pytest


def test_pp_candidate_list():
    with pytest.raises(ProfileError, match="All candidates must be unique."):
        PreferenceProfile(
            ballots=(Ballot(ranking=(frozenset(), frozenset())),),
            candidates=["Peter", "Peter"],
        )

    with pytest.raises(ProfileError, match="Candidate Chris found in ballot "):
        PreferenceProfile(
            ballots=(Ballot(ranking=(frozenset({"Chris"}), frozenset())),),
            candidates=["Peter"],
        )

    with pytest.raises(
        ProfileError,
        match="Candidate Ranking_0 must not share name with"
        " ranking columns: Ranking_i.",
    ):
        PreferenceProfile(
            ballots=(Ballot(ranking=(frozenset({"Ranking_0"}), frozenset())),),
            candidates=["Peter", "Ranking_0"],
        )


def test_pp_excede_ranking_length():
    with pytest.raises(
        ProfileError,
        match="Max ballot length 1 given but ",
    ):
        PreferenceProfile(
            ballots=(Ballot(ranking=(frozenset({"A"}), frozenset({"B"}))),),
            max_ranking_length=1,
        )


def test_pp_contains_ranking():
    with pytest.raises(ProfileError, match="but contains_rankings is set to False."):
        PreferenceProfile(
            ballots=(Ballot(ranking=(frozenset({"A"}),)),), contains_rankings=False
        )

    with pytest.raises(
        ProfileError,
        match="contains_rankings is True but we found no ballots with rankings.",
    ):
        PreferenceProfile(ballots=(Ballot(),), contains_rankings=True)


def test_pp_contains_scores():
    with pytest.raises(ProfileError, match="but contains_scores is set to False."):
        PreferenceProfile(ballots=(Ballot(scores={"A": 2}),), contains_scores=False)

    with pytest.raises(
        ProfileError,
        match="contains_scores is True but we found no ballots with scores.",
    ):
        PreferenceProfile(ballots=(Ballot(),), contains_scores=True)
