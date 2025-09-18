from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile, ProfileError
import pytest


def test_init():
    empty_profile = RankProfile()
    assert empty_profile.ballots == tuple()
    assert not empty_profile.candidates
    assert not empty_profile.candidates_cast
    assert not empty_profile.total_ballot_wt
    assert not empty_profile.num_ballots
    assert empty_profile.max_ranking_length == 0


def test_unique_cands_validator():
    with pytest.raises(ProfileError, match="All candidates must be unique."):
        RankProfile(candidates=("A", "A", "B"))

    RankProfile(candidates=("A", "B"))


def test_strip_whitespace():
    pp = RankProfile(candidates=("A ", " B", " C "))

    assert pp.candidates == ("A", "B", "C")


def test_RankBallots_frozen():
    p = RankProfile(ballots=[RankBallot()])
    b_list = p.ballots

    assert b_list == (RankBallot(),)

    with pytest.raises(
        AttributeError,
        match="Cannot modify frozen instance: tried to set 'ballots'",
    ):
        p.ballots = (RankBallot(weight=5),)


def test_candidates_frozen():
    profile_no_cands = RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}]),
            RankBallot(ranking=[{"C"}, {"B"}]),
        ]
    )
    assert set(profile_no_cands.candidates) == set(["A", "B", "C"])
    assert set(profile_no_cands.candidates_cast) == set(["A", "B", "C"])

    with pytest.raises(
        AttributeError, match="Cannot modify frozen instance: tried to set 'candidates'"
    ):
        profile_no_cands.candidates = tuple()

    with pytest.raises(
        AttributeError,
        match="Cannot modify frozen instance: tried to set 'candidates_cast'",
    ):
        profile_no_cands.candidates_cast = tuple()


def test_get_candidates_received_votes():
    profile_w_cands = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A"}, {"B"}]),
            RankBallot(ranking=[{"C"}, {"B"}]),
        ),
        candidates=("A", "B", "C", "D", "E"),
    )
    vote_cands = profile_w_cands.candidates_cast
    all_cands = profile_w_cands.candidates

    assert set(all_cands) == {"A", "B", "C", "D", "E"}
    assert set(vote_cands) == {
        "A",
        "B",
        "C",
    }
