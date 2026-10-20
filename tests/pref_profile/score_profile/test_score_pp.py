from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from votekit.ballot import ScoreBallot
from votekit.exceptions import ProfileError
from votekit.pref_profile import ScoreProfile
from votekit.pref_profile.utils import convert_row_to_score_ballot


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


def test_tilde_candidates_in_profile_raises_error():
    with pytest.raises(ValueError, match="Candidate '~' found in ScoreProfile.candidates"):
        ScoreProfile(
            ballots=[
                ScoreBallot(scores={1: 1}),
            ],
            candidates=[1, "~"],
        )


def test_candidate_name_with_colon_in_profile_raises_error():
    with pytest.raises(ValueError, match="':' found in ScoreProfile.candidates"):
        ScoreProfile(
            ballots=[
                ScoreBallot(scores={1: 1}),
            ],
            candidates=[1, "A:B"],
        )


def test_non_int_str_candidate_in_profile_raises_error():
    with pytest.raises(
        TypeError, match=r"Non-string/integer candidate\(s\) found in ScoreProfile.candidates"
    ):
        ScoreProfile(
            ballots=[
                ScoreBallot(scores={1: 1}),
            ],
            candidates=[1, 1.0],  # type: ignore[arg-type]
        )


def test_negative_int_candidate_in_profile_raises_error():
    with pytest.raises(
        ValueError, match=r"Negative integer candidate\(s\) found in ScoreProfile.candidates"
    ):
        ScoreProfile(
            ballots=[
                ScoreBallot(scores={1: 1}),
            ],
            candidates=[1, -1],
        )


def test_bool_candidate_in_profile_raises_error():
    with pytest.raises(TypeError, match=r"Boolean candidate\(s\) found in ScoreProfile.candidates"):
        ScoreProfile(
            ballots=[
                ScoreBallot(scores={1: 1}),
            ],
            candidates=[1, True],
        )


def test_tilde_candidate_cast_in_profile_raises_error():
    data = {
        1: [
            1,
            np.nan,
            np.nan,
        ],
        "~": [
            2,
            np.nan,
            np.nan,
        ],
        "Voter Set": [
            set(),
            {"Chris"},
            set(),
        ],
        "Weight": [
            2.0,
            1.0,
            1.0,
        ],
    }
    invalid_cand_df = pd.DataFrame(data)
    invalid_cand_df.index.name = "Ballot Index"
    with pytest.raises(ValueError, match="Candidate '~' found in ScoreProfile.candidates_cast"):
        ScoreProfile(
            df=invalid_cand_df,
            candidates=[
                1,
            ],
        )


def test_candidate_cast_with_colon_in_profile_raises_error():
    data = {
        1: [
            1,
            np.nan,
            np.nan,
        ],
        "A:B": [
            2,
            np.nan,
            np.nan,
        ],
        "Voter Set": [
            set(),
            {"Chris"},
            set(),
        ],
        "Weight": [
            2.0,
            1.0,
            1.0,
        ],
    }
    invalid_cand_df = pd.DataFrame(data)
    invalid_cand_df.index.name = "Ballot Index"
    with pytest.raises(ValueError, match="':' found in ScoreProfile.candidates_cast"):
        ScoreProfile(
            df=invalid_cand_df,
            candidates=[
                1,
            ],
        )


def test_non_int_str_candidate_cast_in_profile_raises_error():
    data = {
        1: [
            1,
            np.nan,
            np.nan,
        ],
        2.0: [
            2,
            np.nan,
            np.nan,
        ],
        "Voter Set": [
            set(),
            {"Chris"},
            set(),
        ],
        "Weight": [
            2.0,
            1.0,
            1.0,
        ],
    }
    invalid_cand_df = pd.DataFrame(data)
    invalid_cand_df.index.name = "Ballot Index"
    print(invalid_cand_df)
    with pytest.raises(
        TypeError, match=r"Non-string/integer candidate\(s\) found in ScoreProfile.candidates_cast"
    ):
        ScoreProfile(
            df=invalid_cand_df,
            candidates=[
                1,
            ],
        )


def test_negative_int_candidate_cast_in_profile_raises_error():
    data = {
        1: [
            1,
            np.nan,
            np.nan,
        ],
        -1: [
            2,
            np.nan,
            np.nan,
        ],
        "Voter Set": [
            set(),
            {"Chris"},
            set(),
        ],
        "Weight": [
            2.0,
            1.0,
            1.0,
        ],
    }
    invalid_cand_df = pd.DataFrame(data)
    invalid_cand_df.index.name = "Ballot Index"
    with pytest.raises(
        ValueError, match=r"Negative integer candidate\(s\) found in ScoreProfile.candidates_cast"
    ):
        ScoreProfile(
            df=invalid_cand_df,
            candidates=[
                1,
            ],
        )


def test_bool_candidate_cast_in_profile_raises_error():
    data = {
        1: [
            1,
            np.nan,
            np.nan,
        ],
        False: [
            2,
            np.nan,
            np.nan,
        ],
        "Voter Set": [
            set(),
            {"Chris"},
            set(),
        ],
        "Weight": [
            2.0,
            1.0,
            1.0,
        ],
    }
    invalid_cand_df = pd.DataFrame(data)
    invalid_cand_df.index.name = "Ballot Index"
    with pytest.raises(
        TypeError, match=r"Boolean candidate\(s\) found in ScoreProfile.candidates_cast"
    ):
        ScoreProfile(
            df=invalid_cand_df,
            candidates=[
                1,
            ],
        )


@pytest.mark.parametrize("exact", [False, True])
def test_ballot_materialization_reads_arrays_and_preserves_values(
    exact: bool, forbid_dataframe_conversion: Callable[[], None]
) -> None:
    weight = Fraction(2, 3) if exact else 0.5
    score_df = pd.DataFrame(
        {
            "A": [0.0, np.nan, 2.0],
            1: [3.0, np.nan, 0.0],
            "zero": [np.nan, np.nan, 9.0],
            "Weight": [weight, weight, 0],
            "Voter Set": [frozenset({"v"}), frozenset(), frozenset()],
        }
    )
    score_df.index.name = "Ballot Index"
    profile = ScoreProfile(df=score_df, candidates=(1, "A"))
    expected = tuple(
        convert_row_to_score_ballot(row, profile.candidates)
        for _, row in profile.to_df().iterrows()
    )

    forbid_dataframe_conversion()

    assert profile.ballots == expected
    assert profile.ballots is profile.ballots
    assert type(profile.ballots[0].weight) is type(weight)


def test_empty_ballots_need_no_dataframe_conversion(
    forbid_dataframe_conversion: Callable[[], None],
) -> None:
    profile = ScoreProfile()

    forbid_dataframe_conversion()

    assert profile.ballots == ()


def test_get_ballot_uses_positions_and_does_not_materialize_other_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ballots = [
        ScoreBallot(scores={"A": 1}),
        ScoreBallot(scores={"B": 2}, weight=Fraction(2, 3), voter_set={"v"}),
    ]
    profile = ScoreProfile(ballots=ballots)
    source = profile.to_df()
    source.index = pd.Index([7, 7], name="Ballot Index")
    profile = ScoreProfile(df=source, candidates=("A", "B"), max_ranking_length=1)

    def reject_materialization(*args: object, **kwargs: object) -> None:
        raise AssertionError("Single-row access must not materialize the profile")

    monkeypatch.setattr(ScoreProfile, "ballots", property(reject_materialization))
    monkeypatch.setattr(ScoreProfile, "to_df", reject_materialization)

    assert profile.get_ballot(1) == ballots[1]
    assert profile.get_ballot(-1) == ballots[1]
    assert "ballots" not in profile.__dict__

    with pytest.raises(IndexError):
        profile.get_ballot(2)


def test_iter_ballots_is_lazy_and_leaves_cache_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    ballots = [ScoreBallot(scores={"A": 1}), ScoreBallot(weight=0)]
    profile = ScoreProfile(ballots=ballots)

    requested_rows: list[int] = []
    get_ballot = profile.get_ballot

    def record_row(self: ScoreProfile, row: int) -> ScoreBallot:
        assert self is profile

        requested_rows.append(row)

        return get_ballot(row)

    monkeypatch.setattr(ScoreProfile, "get_ballot", record_row)

    iterator = profile.iter_ballots()

    assert requested_rows == []

    assert next(iterator) == ballots[0]
    assert requested_rows == [0]
    assert "ballots" not in profile.__dict__

    assert list(iterator) == [ballots[1]]
    assert requested_rows == [0, 1]
    assert "ballots" not in profile.__dict__

    assert profile.ballots == tuple(ballots)
    assert profile.ballots is profile.ballots

    assert next(profile.iter_ballots()) is not profile.ballots[0]
