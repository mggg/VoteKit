from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import pandas as pd
import pytest

from votekit.ballot import RankBallot
from votekit.pref_profile import ProfileError, RankProfile
from votekit.pref_profile.utils import convert_row_to_rank_ballot


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


def test_tilde_candidates_in_profile_raises_error():
    with pytest.raises(ValueError, match="Candidate '~' found in RankProfile.candidates"):
        RankProfile(
            ballots=[RankBallot(ranking=["A"])],
            candidates=["A", "~"],
        )


def test_candidate_name_with_colon_in_profile_raises_error():
    with pytest.raises(ValueError, match="':' found in RankProfile.candidates"):
        RankProfile(
            ballots=[RankBallot(ranking=["A"])],
            candidates=["A", "A:B"],
        )


def test_non_int_str_candidate_in_profile_raises_error():
    with pytest.raises(
        TypeError, match=r"Non-string/integer candidate\(s\) found in RankProfile.candidates"
    ):
        RankProfile(
            ballots=[RankBallot(ranking=["A"])],
            candidates=["A", 1.0],  # type: ignore[arg-type]
        )


def test_negative_int_candidate_in_profile_raises_error():
    with pytest.raises(
        ValueError, match=r"Negative integer candidate\(s\) found in RankProfile.candidates"
    ):
        RankProfile(
            ballots=[RankBallot(ranking=["A"])],
            candidates=["A", -1],
        )


def test_bool_candidate_in_profile_raises_error():
    with pytest.raises(TypeError, match=r"Boolean candidate\(s\) found in RankProfile.candidates"):
        RankProfile(
            ballots=[RankBallot(ranking=["A"])],
            candidates=["A", False],
        )


def test_candidate_cast_with_colon_in_profile_raises_error():
    data = {
        "Ranking_1": [frozenset({"A"}), frozenset({"A"}), frozenset({"A:B"})],
        "Ranking_2": [
            frozenset(),
            frozenset(),
            frozenset({"A"}),
        ],
        "Voter Set": [set(), {"Chris"}, set()],
        "Weight": [2.0, 0.0, 1.0],
    }
    invalid_cand_df = pd.DataFrame(data)
    invalid_cand_df.index.name = "Ballot Index"
    with pytest.raises(ValueError, match="':' found in RankProfile.candidates_cast"):
        RankProfile(
            df=invalid_cand_df,
            candidates=[
                "A",
            ],
            max_ranking_length=2,
        )


def test_non_int_str_candidate_cast_in_profile_raises_error():
    data = {
        "Ranking_1": [frozenset({"A"}), frozenset({"A"}), frozenset({1.0})],
        "Ranking_2": [
            frozenset(),
            frozenset(),
            frozenset({"A"}),
        ],
        "Voter Set": [set(), {"Chris"}, set()],
        "Weight": [2.0, 0.0, 1.0],
    }
    invalid_cand_df = pd.DataFrame(data)
    invalid_cand_df.index.name = "Ballot Index"
    with pytest.raises(
        TypeError, match=r"Non-string/integer candidate\(s\) found in RankProfile.candidates_cast"
    ):
        RankProfile(
            df=invalid_cand_df,
            candidates=[
                "A",
            ],
            max_ranking_length=2,
        )


def test_negative_int_candidate_cast_in_profile_raises_error():
    data = {
        "Ranking_1": [frozenset({"A"}), frozenset({"A"}), frozenset({-1})],
        "Ranking_2": [
            frozenset(),
            frozenset(),
            frozenset({"A"}),
        ],
        "Voter Set": [set(), {"Chris"}, set()],
        "Weight": [2.0, 0.0, 1.0],
    }
    invalid_cand_df = pd.DataFrame(data)
    invalid_cand_df.index.name = "Ballot Index"
    with pytest.raises(
        ValueError, match=r"Negative integer candidate\(s\) found in RankProfile.candidates_cast"
    ):
        RankProfile(
            df=invalid_cand_df,
            candidates=[
                "A",
            ],
            max_ranking_length=2,
        )


def test_bool_candidate_cast_in_profile_raises_error():
    data = {
        "Ranking_1": [frozenset({"A"}), frozenset({"A"}), frozenset({False})],
        "Ranking_2": [
            frozenset(),
            frozenset(),
            frozenset({"A"}),
        ],
        "Voter Set": [set(), {"Chris"}, set()],
        "Weight": [2.0, 0.0, 1.0],
    }
    invalid_cand_df = pd.DataFrame(data)
    invalid_cand_df.index.name = "Ballot Index"
    with pytest.raises(
        TypeError, match=r"Boolean candidate\(s\) found in RankProfile.candidates_cast"
    ):
        RankProfile(
            df=invalid_cand_df,
            candidates=[
                "A",
            ],
            max_ranking_length=2,
        )


@pytest.mark.parametrize("exact", [False, True])
def test_ballot_materialization_reads_arrays_and_preserves_values(
    exact: bool, forbid_dataframe_conversion: Callable[[], None]
) -> None:
    weight = Fraction(2, 3) if exact else 0.5
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=[{"A", 1}, set(), {"A"}], weight=weight, voter_set={"v"}),
            RankBallot(weight=weight),
            RankBallot(ranking=[{"zero"}], weight=0),
        ],
        candidates=("A", 1, "zero"),
        max_ranking_length=3,
    )
    expected = tuple(convert_row_to_rank_ballot(row, 3) for _, row in profile.to_df().iterrows())

    forbid_dataframe_conversion()

    assert profile.ballots == expected
    assert profile.ballots is profile.ballots
    assert type(profile.ballots[0].weight) is type(weight)


def test_empty_ballots_need_no_dataframe_conversion(
    forbid_dataframe_conversion: Callable[[], None],
) -> None:
    profile = RankProfile()

    forbid_dataframe_conversion()

    assert profile.ballots == ()


def test_get_ballot_uses_positions_and_does_not_materialize_other_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ballots = [
        RankBallot(ranking=[{"A"}]),
        RankBallot(ranking=[{"B"}], weight=Fraction(2, 3), voter_set={"v"}),
    ]
    profile = RankProfile(ballots=ballots)
    source = profile.to_df()
    source.index = pd.Index([7, 7], name="Ballot Index")
    profile = RankProfile(df=source, candidates=("A", "B"), max_ranking_length=1)

    def reject_materialization(*args: object, **kwargs: object) -> None:
        raise AssertionError("Single-row access must not materialize the profile")

    monkeypatch.setattr(RankProfile, "ballots", property(reject_materialization))
    monkeypatch.setattr(RankProfile, "to_df", reject_materialization)

    assert profile.get_ballot(1) == ballots[1]
    assert profile.get_ballot(-1) == ballots[1]
    assert "ballots" not in profile.__dict__

    with pytest.raises(IndexError):
        profile.get_ballot(2)


def test_iter_ballots_is_lazy_and_leaves_cache_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    ballots = [RankBallot(ranking=[{"A"}]), RankBallot(weight=0)]
    profile = RankProfile(ballots=ballots)

    requested_rows: list[int] = []
    get_ballot = profile.get_ballot

    def record_row(self: RankProfile, row: int) -> RankBallot:
        assert self is profile

        requested_rows.append(row)

        return get_ballot(row)

    monkeypatch.setattr(RankProfile, "get_ballot", record_row)

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


def test_get_ballot_preserves_empty_positions_before_end_sentinel() -> None:
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, set(), {"B"}])],
        max_ranking_length=4,
    )
    expected = (frozenset({"A"}), frozenset(), frozenset({"B"}))

    assert profile.get_ballot(0).ranking == expected
