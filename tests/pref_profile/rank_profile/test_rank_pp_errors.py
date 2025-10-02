from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile, ProfileError
import pytest


def test_pp_candidate_list():
    with pytest.raises(ProfileError, match="All candidates must be unique."):
        RankProfile(
            ballots=(RankBallot(ranking=(frozenset(), frozenset())),),
            candidates=["Peter", "Peter"],
        )

    with pytest.raises(ProfileError, match="Candidate Chris found in ballot "):
        RankProfile(
            ballots=(RankBallot(ranking=(frozenset({"Chris"}), frozenset())),),
            candidates=["Peter"],
        )

    with pytest.raises(
        ProfileError,
        match="Ranking_0 is a name reserved for ranking columns, it cannot be used as a candidate"
        " name.",
    ):
        RankProfile(
            ballots=(RankBallot(ranking=(frozenset({"Ranking_0"}), frozenset())),),
            candidates=["Peter", "Ranking_0"],
        )


def test_pp_excede_ranking_length():
    with pytest.raises(
        ProfileError,
        match="Max ranking length 1 given but ",
    ):
        RankProfile(
            ballots=(RankBallot(ranking=(frozenset({"A"}), frozenset({"B"}))),),
            max_ranking_length=1,
        )
