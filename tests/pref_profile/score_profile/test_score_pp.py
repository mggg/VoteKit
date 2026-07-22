import pytest

from votekit.ballot import ScoreBallot
from votekit.exceptions import ProfileError
from votekit.pref_profile import ScoreProfile


def test_init():
    empty_profile = ScoreProfile()
    print(type(empty_profile))
    assert empty_profile.ballots == tuple()
    assert not empty_profile.candidates
    assert not empty_profile.candidates_cast
    assert not empty_profile.total_ballot_wt
    assert not empty_profile.num_ballots


def test_unique_cands_validator():
    with pytest.raises(ProfileError, match="All candidates must be unique."):
        ScoreProfile(candidates=("A", "A", "B"))


def test_strip_whitespace():
    pp = ScoreProfile(candidates=("A ", " B", " C "))

    assert pp.candidates == ("A", "B", "C")


def test_ballots_frozen():
    p = ScoreProfile(ballots=[ScoreBallot()])
    b_list = p.ballots

    assert b_list == (ScoreBallot(),)

    with pytest.raises(
        AttributeError,
        match="Cannot modify frozen instance: tried to set 'ballots'",
    ):
        p.ballots = (ScoreBallot(weight=5),)


def test_candidates_frozen():
    profile_no_cands = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 4}),
            ScoreBallot(scores={"B": 4}),
            ScoreBallot(scores={"C": 4}),
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
    profile_w_cands = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 4}),
            ScoreBallot(scores={"B": 4}),
            ScoreBallot(scores={"C": 4}),
        ],
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
    profile_int_cands = ScoreProfile(
        ballots=[
            ScoreBallot(scores={1: 4}),
            ScoreBallot(scores={2: 4}),
            ScoreBallot(scores={3: 4}),
        ]
    )

    assert set(profile_int_cands.candidates) == set([1, 2, 3])
    assert set(profile_int_cands.candidates_cast) == set([1, 2, 3])


def test_str_int_mix_candidates():
    profile_int_cands = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 4}),
            ScoreBallot(scores={2: 4}),
            ScoreBallot(scores={3: 4}),
        ]
    )

    assert set(profile_int_cands.candidates) == set(["A", 2, 3])
    assert set(profile_int_cands.candidates_cast) == set(["A", 2, 3])


def test_str_int_mix_candidates_in_profile_gives_warning():
    with pytest.warns(UserWarning, match="will be treated as separate candidates"):
        profile_int_cands = ScoreProfile(
            ballots=[
                ScoreBallot(scores={"2": 4}),
                ScoreBallot(scores={2: 4}),
                ScoreBallot(scores={3: 4}),
            ]
        )

    assert set(profile_int_cands.candidates) == set(["2", 2, 3])
    assert set(profile_int_cands.candidates_cast) == set(["2", 2, 3])


def test_tilda_candidates_in_profile_raises_error():
    with pytest.raises(ValueError, match="Candidate '~' found in profile's candidates"):
        ScoreProfile(
            ballots=[
                ScoreBallot(scores={1: 1}),
            ],
            candidates=[1, "~"],
        )


def test_candidate_name_with_colon_in_profile_raises_error():
    with pytest.raises(ValueError, match="':' found in profile's candidates"):
        ScoreProfile(
            ballots=[
                ScoreBallot(scores={1: 1}),
            ],
            candidates=[1, "A:B"],
        )


def test_non_int_str_candidate_in_profile_raises_error():
    with pytest.raises(
        TypeError, match="Non-string/integer candidate(s) found in profile's candidates"
    ):
        ScoreProfile(  # type: ignore[arg-type]
            ballots=[
                ScoreBallot(scores={1: 1}),
            ],
            candidates=[1, 1.0],  # type: ignore[arg-type]
        )


def test_negative_int_candidate_in_profile_raises_error():
    with pytest.raises(
        ValueError, match=r"Negative integer candidate\(s\) found in profile's candidates"
    ):
        ScoreProfile(
            ballots=[
                ScoreBallot(scores={1: 1}),
            ],
            candidates=[1, -1],
        )


def test_bool_candidate_in_profile_raises_error():
    with pytest.raises(TypeError, match=r"Boolean candidate\(s\) found in profile's candidates"):
        ScoreProfile(
            ballots=[
                ScoreBallot(scores={1: 1}),
            ],
            candidates=[1, True],
        )
