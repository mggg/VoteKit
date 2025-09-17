from votekit.ballot import ScoreBallot
from votekit.pref_profile import ScoreProfile, ProfileError
import pytest


def test_pp_candidate_list():
    with pytest.raises(ProfileError, match="All candidates must be unique."):
        ScoreProfile(
            ballots=(ScoreBallot(),),
            candidates=["Peter", "Peter"],
        )

    with pytest.raises(ProfileError, match="Candidate Chris found in ballot "):
        ScoreProfile(
            ballots=(ScoreBallot(scores={"Chris": 1}),),
            candidates=["Peter"],
        )

    with pytest.raises(
        ProfileError,
        match="Ranking_0 is a name reserved for ranking columns, it cannot be used as a candidate"
        " name.",
    ):
        ScoreProfile(
            ballots=(ScoreBallot(scores={"Ranking_0": 1}),),
            candidates=["Peter", "Ranking_0"],
        )
