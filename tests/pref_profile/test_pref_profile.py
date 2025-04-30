from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
import pytest
import dataclasses


def test_init():
    empty_profile = PreferenceProfile()
    assert empty_profile.ballots == ()
    assert not empty_profile.candidates
    assert not empty_profile.candidates_cast
    assert not empty_profile.total_ballot_wt
    assert not empty_profile.num_ballots
    assert empty_profile.max_ranking_length == 0


def test_unique_cands_validator():
    with pytest.raises(ValueError, match="All candidates must be unique."):
        PreferenceProfile(candidates=("A", "A", "B"))

    PreferenceProfile(candidates=("A", "B"))


def test_strip_whitespace():
    pp = PreferenceProfile(candidates=("A ", " B", " C "))

    assert pp.candidates == ("A", "B", "C")


def test_ballots_frozen():
    p = PreferenceProfile(ballots=[Ballot()])
    b_list = p.ballots

    assert b_list == (Ballot(),)

    with pytest.raises(
        dataclasses.FrozenInstanceError, match="cannot assign to field 'ballots'"
    ):
        p.ballots = (Ballot(weight=5),)


def test_candidates_frozen():
    profile_no_cands = PreferenceProfile(
        ballots=[Ballot(ranking=[{"A"}, {"B"}]), Ballot(ranking=[{"C"}, {"B"}])]
    )
    assert set(profile_no_cands.candidates) == set(["A", "B", "C"])
    assert set(profile_no_cands.candidates_cast) == set(["A", "B", "C"])

    with pytest.raises(
        dataclasses.FrozenInstanceError, match="cannot assign to field 'candidates'"
    ):
        profile_no_cands.candidates = tuple()

    with pytest.raises(
        dataclasses.FrozenInstanceError,
        match="cannot assign to field 'candidates_cast'",
    ):
        profile_no_cands.candidates_cast = tuple()


def test_get_candidates_received_votes():
    profile_w_cands = PreferenceProfile(
        ballots=(
            Ballot(ranking=[{"A"}, {"B"}]),
            Ballot(ranking=[{"C"}, {"B"}]),
            Ballot(scores={"A": 4, "E": 4}),
        ),
        candidates=("A", "B", "C", "D", "E"),
    )
    vote_cands = profile_w_cands.candidates_cast
    all_cands = profile_w_cands.candidates

    assert set(all_cands) == {"A", "B", "C", "D", "E"}
    assert set(vote_cands) == {"A", "B", "C", "E"}
