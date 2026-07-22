import pytest

from votekit.ballot import RankBallot
from votekit.pref_profile import ProfileError, RankProfile


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


def test_int_only_candidates():
    profile_w_int_cands = RankProfile(
        ballots=(
            RankBallot(ranking=[{2}, {1}]),
            RankBallot(ranking=[{3}, {4}]),
            RankBallot(ranking=[{1, 2}]),
        ),
    )
    vote_cands = profile_w_int_cands.candidates_cast
    all_cands = profile_w_int_cands.candidates

    assert set(vote_cands) == {1, 2, 3, 4}
    assert set(all_cands) == {1, 2, 3, 4}


def test_str_int_mix_candidates():
    profile_w_mix_cands = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A"}, {1}]),
            RankBallot(ranking=[{"C"}, {"B"}]),
            RankBallot(ranking=[{1, 2}]),
        ),
    )
    vote_cands = profile_w_mix_cands.candidates_cast
    all_cands = profile_w_mix_cands.candidates

    assert set(vote_cands) == {"A", "B", "C", 1, 2}
    assert set(all_cands) == {"A", "B", "C", 1, 2}


def test_equivalent_str_int_cands_in_profile_gives_warning():
    with pytest.warns(UserWarning, match="will be treated as separate candidates"):
        profile_w_mix_cands = RankProfile(
            ballots=(
                RankBallot(ranking=[{"A"}, {1}]),
                RankBallot(ranking=[{"1"}, {"B"}]),
                RankBallot(ranking=[{1, 2}]),
            ),
        )
    vote_cands = profile_w_mix_cands.candidates_cast
    all_cands = profile_w_mix_cands.candidates

    assert set(vote_cands) == {"A", "B", "1", 1, 2}
    assert set(all_cands) == {"A", "B", "1", 1, 2}


def test_tilda_candidates_in_profile_raises_error():
    with pytest.raises(ValueError, match="Candidate '~' found in profile's candidates"):
        RankProfile(
            ballots=[RankBallot(ranking=["A"])],
            candidates=["A", "~"],
        )


def test_candidate_name_with_colon_in_profile_raises_error():
    with pytest.raises(ValueError, match="':' found in profile's candidates"):
        RankProfile(
            ballots=[RankBallot(ranking=["A"])],
            candidates=["A", "A:B"],
        )


def test_non_int_str_candidate_in_profile_raises_error():
    with pytest.raises(
        TypeError, match="Non-string/integer candidate(s) found in profile's candidates"
    ):
        RankProfile(  # type: ignore[arg-type]
            ballots=[RankBallot(ranking=["A"])],
            candidates=["A", 1.0],  # type: ignore[arg-type]
        )


def test_negative_int_candidate_in_profile_raises_error():
    with pytest.raises(
        ValueError, match=r"Negative integer candidate\(s\) found in profile's candidates"
    ):
        RankProfile(
            ballots=[RankBallot(ranking=["A"])],
            candidates=["A", -1],
        )


def test_bool_candidate_in_profile_raises_error():
    with pytest.raises(TypeError, match=r"Boolean candidate\(s\) found in profile's candidates"):
        RankProfile(
            ballots=[RankBallot(ranking=["A"])],
            candidates=["A", False],
        )
