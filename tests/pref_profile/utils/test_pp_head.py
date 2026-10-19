from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.pref_profile import PreferenceProfile, RankProfile, ScoreProfile, profile_df_head
from votekit.pref_profile.profile_matrix import RankMatrix, ScoreMatrix


def test_pp_df_head_rankings():
    ballots_rankings = [
        RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
        RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
        RankBallot(),
        RankBallot(weight=0),
    ]
    pp = PreferenceProfile(ballots=ballots_rankings)
    data = {
        "Ranking_1": [
            frozenset({"A"}),
            frozenset({"A", "B"}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Ranking_2": [frozenset({"B"}), frozenset(), frozenset("~"), frozenset("~")],
        "Ranking_3": [
            frozenset({"C"}),
            frozenset({"D"}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [2.0, 1.0, 1.0, 0.0],
        "Percent": [
            f"{0.5:.1%}",
            f"{0.25:.1%}",
            f"{0.25:.1%}",
            f"{0:.1%}",
        ],
    }
    true_df = pd.DataFrame(data)
    true_df.index = [1, 0, 2, 3]
    true_df.index.name = "Ballot Index"
    true_df.loc["Total"] = [""] * 4 + [4, f"{1:.1%}"]

    assert profile_df_head(pp, n=4, percents=True, totals=True).equals(true_df)


def test_pp_df_head_scores():
    ballots_scores = [
        ScoreBallot(scores={"D": 2, "E": 1}, voter_set={"Chris"}),
        ScoreBallot(
            weight=2,
            scores={
                "A": 1,
                "B": 2,
            },
        ),
        ScoreBallot(),
        ScoreBallot(weight=0),
    ]
    pp = PreferenceProfile(ballots=ballots_scores)
    data = {
        "A": [
            1,
            np.nan,
            np.nan,
            np.nan,
        ],
        "B": [
            2,
            np.nan,
            np.nan,
            np.nan,
        ],
        "D": [np.nan, 2, np.nan, np.nan],
        "E": [np.nan, 1, np.nan, np.nan],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [2.0, 1.0, 1.0, 0.0],
        "Percent": [
            f"{0.5:.2%}",
            f"{0.25:.2%}",
            f"{0.25:.2%}",
            f"{0:.2%}",
        ],
    }
    true_df = pd.DataFrame(data)
    true_df.index = [1, 0, 2, 3]
    true_df.index.name = "Ballot Index"
    true_df.loc["Total"] = [""] * 5 + [4, f"{1:.2%}"]

    assert profile_df_head(pp, n=4, percents=True, totals=True, n_decimals=2).equals(true_df)


@pytest.mark.parametrize("profile_type", [RankProfile, ScoreProfile])
@pytest.mark.parametrize("n", [-2, 0, 2, 10])
@pytest.mark.parametrize("sort_by_weight", [False, True])
def test_head_converts_only_selected_rows_to_dataframe(
    profile_type: type[RankProfile] | type[ScoreProfile],
    n: int,
    sort_by_weight: bool,
    monkeypatch: pytest.MonkeyPatch,
    unexpected_dataframe_conversion: Callable[..., None],
) -> None:
    if profile_type is RankProfile:
        ballots = [RankBallot(ranking=[{c}], weight=w) for c, w in zip("ABCD", [2, 1, 2, 0])]
    else:
        ballots = [ScoreBallot(scores={c: 1}, weight=w) for c, w in zip("ABCD", [2, 1, 2, 0])]

    source = profile_type(ballots=ballots).to_df()
    source.index = pd.Index([7, 7, 9, 3], name="Ballot Index")
    profile = profile_type(df=source, candidates=("A", "B", "C", "D"), max_ranking_length=1)
    expected = (
        source.sort_values("Weight", ascending=False, kind="stable") if sort_by_weight else source
    )
    expected = expected.head(n)
    matrix_to_df = profile._matrix.to_df

    def convert_selected_rows_to_dataframe(
        matrix: RankMatrix | ScoreMatrix, *, rows: np.ndarray | None = None
    ) -> pd.DataFrame:
        assert matrix is profile._matrix
        assert rows is not None

        return matrix_to_df(rows=rows)

    monkeypatch.setattr(profile_type, "to_df", unexpected_dataframe_conversion)
    monkeypatch.setattr(type(profile._matrix), "to_df", convert_selected_rows_to_dataframe)

    pd.testing.assert_frame_equal(
        profile_df_head(profile, n, sort_by_weight=sort_by_weight), expected
    )


@pytest.mark.parametrize("profile_type", [RankProfile, ScoreProfile])
def test_head_empty_profile_matches_to_df(
    profile_type: type[RankProfile] | type[ScoreProfile],
    monkeypatch: pytest.MonkeyPatch,
    unexpected_dataframe_conversion: Callable[..., None],
) -> None:
    profile = profile_type(candidates=("A",))
    expected = profile.to_df()

    with monkeypatch.context() as patch:
        patch.setattr(profile_type, "to_df", unexpected_dataframe_conversion)

        actual = profile_df_head(profile, 5)

    pd.testing.assert_frame_equal(actual, expected)
