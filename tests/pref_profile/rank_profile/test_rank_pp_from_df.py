import pandas as pd
import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.pref_profile import ProfileError, RankProfile

profile = RankProfile(
    ballots=(
        RankBallot(ranking=[{"A"}, {"B"}]),
        RankBallot(ranking=[{"C"}, {"B"}]),
        RankBallot(ranking=[{"A"}, {"B"}]),
        RankBallot(ranking=[{"D"}], weight=0),
    ),
    candidates=("A", "B", "C", "D", "E"),
    max_ranking_length=3,
)
df = profile.df


def test_from_df():
    new_profile = RankProfile(
        df=df,
        candidates=profile.candidates,
        max_ranking_length=profile.max_ranking_length,
    )

    assert new_profile == profile
    assert set(new_profile.candidates_cast) == {
        "A",
        "B",
        "C",
    }


def test_from_df_init_errors():
    with pytest.raises(
        ProfileError,
        match="Cannot pass a dataframe and a ballot list to profile init method. Must pick one.",
    ):
        RankProfile(df=df, ballots=(RankBallot(ranking=[{"Chris"}]),))

    with pytest.raises(
        ProfileError,
        match="max_ranking_length must be provided and be non-zero.",
    ):
        RankProfile(df=df)

    with pytest.raises(
        ProfileError,
        match="RankProfile requires RankBallots; ballot at index 1 is ScoreBallot.",
    ):
        RankProfile(ballots=(RankBallot(ranking=[{"Chris"}]), ScoreBallot(scores={"Chris": 1})))


def test_from_df_validation_errors():
    with pytest.raises(ProfileError, match="Weight column not in dataframe:"):
        RankProfile(
            df=pd.DataFrame(columns=["Voter Set", "Ranking_1"]),
            max_ranking_length=3,
            candidates=["A"],
        )

    with pytest.raises(ProfileError, match="Voter Set column not in dataframe:"):
        RankProfile(
            df=pd.DataFrame(columns=["Weight", "Ranking_1"]),
            max_ranking_length=3,
            candidates=["A"],
        )

    with pytest.raises(ProfileError, match="Index not named 'Ballot Index':"):
        RankProfile(
            df=pd.DataFrame(columns=["Weight", "Voter Set", "Ranking_1"]),
            max_ranking_length=3,
            candidates=["A"],
        )

    with pytest.raises(ProfileError, match="Ranking column 'Ranking_2' not in dataframe:"):
        df = pd.DataFrame(columns=["Weight", "Voter Set", "Ranking_1", "Ranking_3"])
        df.index.name = "Ballot Index"
        RankProfile(
            df=df,
            max_ranking_length=3,
            candidates=["A"],
        )


@pytest.mark.parametrize("declared_candidates", [(" A ", " unused "), ("A", "unused"), ()])
def test_from_df_normalizes_candidate_identities_without_mutating_input(declared_candidates):
    source = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}]), RankBallot(weight=0, ranking=[{"B"}])]
    ).df.copy()
    source["Ranking_1"] = pd.Series([frozenset({"A", " A "}), frozenset({" B "})])
    original = source.copy(deep=True)

    result = RankProfile(df=source, candidates=declared_candidates, max_ranking_length=1)

    expected_candidates = ("A", "unused") if declared_candidates else ("A",)
    assert result.candidates == result._matrix.candidates == expected_candidates
    assert result.candidates_cast == ("A",)
    assert result.df["Ranking_1"].tolist() == [frozenset({"A"}), frozenset({"B"})]
    assert result.ballots[0].ranking == (frozenset({"A"}),)
    assert result.ballots[1].ranking == (frozenset({"B"}),)
    pd.testing.assert_frame_equal(source, original)


def test_from_df_normalization_preserves_empty_and_tilde_sets():
    source = RankProfile(ballots=[RankBallot(ranking=[{"A"}, set()]), RankBallot()]).df.copy()
    source["Ranking_1"] = pd.Series([frozenset({" A "}), frozenset({"~"})])
    result = RankProfile(df=source, candidates=[" A "], max_ranking_length=2)

    assert result.df["Ranking_1"].tolist() == [frozenset({"A"}), frozenset({"~"})]
    assert result.df["Ranking_2"].tolist() == [frozenset(), frozenset({"~"})]


@pytest.mark.parametrize("weight", [0, 1])
def test_from_df_rejects_padded_tilde_candidate(weight):
    source = RankProfile(ballots=[RankBallot(ranking=[{"A"}], weight=weight)]).df.copy()
    source["Ranking_1"] = pd.Series([frozenset({" ~ "})])

    with pytest.raises(ValueError, match="reserved character"):
        RankProfile(df=source, candidates=["A"], max_ranking_length=1)


def test_from_df_normalization_preserves_integer_and_string_candidates():
    source = RankProfile(ballots=[RankBallot(ranking=[{1}, {"A"}])]).df.copy()
    source["Ranking_2"] = pd.Series([frozenset({" 1 "})])

    with pytest.warns(UserWarning, match="both str and int"):
        result = RankProfile(df=source, candidates=[1, " 1 "], max_ranking_length=2)

    assert result.candidates == result._matrix.candidates == (1, "1")
    with pytest.warns(UserWarning, match="both str and int"):
        assert result.ballots[0].ranking == (frozenset({1}), frozenset({"1"}))


@pytest.mark.parametrize("cell", [" A ", (" A ",)])
def test_from_df_normalization_rejects_non_set_ranking_cells(cell):
    source = RankProfile(ballots=[RankBallot(ranking=[{"A"}])]).df.copy()
    source["Ranking_1"] = pd.Series([cell], dtype=object)

    with pytest.raises(TypeError, match="Ranking cells must contain sets"):
        RankProfile(df=source, candidates=["A"], max_ranking_length=1)
