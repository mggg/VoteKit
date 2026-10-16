import pandas as pd
import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.pref_profile import ProfileError, ScoreProfile

profile = ScoreProfile(
    ballots=(
        ScoreBallot(scores={"A": 4, "E": 4}),
        ScoreBallot(
            scores={"A": 4, "E": 4},
        ),
        ScoreBallot(scores={"D": 3.1}, weight=0),
    ),
    candidates=("A", "B", "C", "D", "E"),
)
df = profile.df


def test_from_df():
    new_profile = ScoreProfile(
        df=df,
        candidates=profile.candidates,
    )

    assert new_profile == profile
    assert set(new_profile.candidates_cast) == {"A", "E"}


def test_from_df_init_errors():
    with pytest.raises(
        ProfileError,
        match="Cannot pass a dataframe and a ballot list to profile init method. Must pick one.",
    ):
        ScoreProfile(df=df, ballots=(ScoreBallot(scores={"Chris": 1}),))

    with pytest.raises(
        ProfileError,
        match="ScoreProfile requires ScoreBallots; ballot at index 0 is RankBallot.",
    ):
        ScoreProfile(ballots=(RankBallot(ranking=[{"Chris"}]), ScoreBallot(scores={"Chris": 1})))

    with pytest.raises(
        ProfileError,
        match="When providing a dataframe and no ballot list to the init method, "
        "candidates must be provided.",
    ):
        ScoreProfile(df=df)


def test_from_df_validation_errors():
    with pytest.raises(ProfileError, match="Weight column not in dataframe:"):
        ScoreProfile(
            df=pd.DataFrame(columns=["Voter Set"]),
            candidates=["A"],
        )

    with pytest.raises(ProfileError, match="Voter Set column not in dataframe:"):
        ScoreProfile(
            df=pd.DataFrame(columns=["Weight"]),
            candidates=["A"],
        )

    with pytest.raises(ProfileError, match="Index not named 'Ballot Index':"):
        ScoreProfile(
            df=pd.DataFrame(columns=["Weight", "Voter Set"]),
            candidates=["A"],
        )

    with pytest.raises(ProfileError, match="Candidate column 'B' not in dataframe:"):
        df = pd.DataFrame(columns=["Weight", "Voter Set", "A", "C"])
        df.index.name = "Ballot Index"
        ScoreProfile(
            df=df,
            candidates=["A", "B", "C"],
        )


@pytest.mark.parametrize("declared_candidates", [(" A ", " unused "), ("A", "unused")])
def test_from_df_normalizes_candidate_identities_without_mutating_input(declared_candidates):
    source = ScoreProfile(
        ballots=[ScoreBallot(scores={"A": 2}), ScoreBallot(scores={"B": 3}, weight=0)],
    ).df.copy()
    source["unused"] = float("nan")
    source = source.rename(columns={"A": " A ", "B": " B ", "unused": " unused "})
    original = source.copy(deep=True)

    result = ScoreProfile(df=source, candidates=declared_candidates)

    assert result.candidates == result._matrix.candidates == ("A", "unused")
    assert result.candidates_cast == ("A",)
    assert result._matrix.score_columns == ("A", "B", "unused")
    assert result.ballots[0].scores == {"A": 2}
    assert result.df["B"].tolist()[1] == 3
    pd.testing.assert_frame_equal(source, original)


@pytest.mark.parametrize("second_column", ["A", " A "])
def test_from_df_rejects_colliding_score_column_names(second_column):
    source = pd.DataFrame(
        [[1.0, 2.0, frozenset(), 1.0]], columns=["A", second_column, "Voter Set", "Weight"]
    )
    source.index.name = "Ballot Index"

    with pytest.raises(ProfileError, match="same identity after stripping whitespace"):
        ScoreProfile(df=source, candidates=["A"])


@pytest.mark.parametrize("column", [" Weight ", " Voter Set "])
def test_from_df_rejects_score_columns_normalizing_to_shared_names(column):
    source = ScoreProfile(ballots=[ScoreBallot(scores={"A": 1})]).df.rename(columns={"A": column})

    with pytest.raises(ProfileError, match="reserved dataframe column names"):
        ScoreProfile(df=source, candidates=[column])


def test_from_df_normalization_preserves_integer_and_string_candidates():
    source = ScoreProfile(ballots=[ScoreBallot(scores={1: 2, "A": 3})]).df.rename(
        columns={"A": " 1 "}
    )

    with pytest.warns(UserWarning, match="both str and int"):
        result = ScoreProfile(df=source, candidates=[1, " 1 "])

    assert result.candidates == result._matrix.candidates == (1, "1")
    with pytest.warns(UserWarning, match="both str and int"):
        assert result.ballots[0].scores == {1: 2, "1": 3}
