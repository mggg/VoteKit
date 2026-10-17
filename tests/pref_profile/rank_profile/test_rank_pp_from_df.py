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
df = profile.to_df()


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
    source = (
        RankProfile(ballots=[RankBallot(ranking=[{"A"}]), RankBallot(weight=0, ranking=[{"B"}])])
        .to_df()
        .copy()
    )
    source["Ranking_1"] = pd.Series([frozenset({"A", " A "}), frozenset({" B "})])
    original = source.copy(deep=True)

    result = RankProfile(df=source, candidates=declared_candidates, max_ranking_length=1)

    expected_candidates = ("A", "unused") if declared_candidates else ("A",)
    assert result.candidates == result._matrix.candidates == expected_candidates
    assert result.candidates_cast == ("A",)
    assert result.to_df()["Ranking_1"].tolist() == [frozenset({"A"}), frozenset({"B"})]
    assert result.ballots[0].ranking == (frozenset({"A"}),)
    assert result.ballots[1].ranking == (frozenset({"B"}),)
    pd.testing.assert_frame_equal(source, original)


def test_from_df_normalization_preserves_empty_and_tilde_sets():
    source = RankProfile(ballots=[RankBallot(ranking=[{"A"}, set()]), RankBallot()]).to_df().copy()
    source["Ranking_1"] = pd.Series([frozenset({" A "}), frozenset({"~"})])
    result = RankProfile(df=source, candidates=[" A "], max_ranking_length=2)

    assert result.to_df()["Ranking_1"].tolist() == [frozenset({"A"}), frozenset({"~"})]
    assert result.to_df()["Ranking_2"].tolist() == [frozenset(), frozenset({"~"})]


@pytest.mark.parametrize("weight", [0, 1])
def test_from_df_rejects_padded_tilde_candidate(weight):
    source = RankProfile(ballots=[RankBallot(ranking=[{"A"}], weight=weight)]).to_df().copy()
    source["Ranking_1"] = pd.Series([frozenset({" ~ "})])

    with pytest.raises(ValueError, match="reserved character"):
        RankProfile(df=source, candidates=["A"], max_ranking_length=1)


def test_from_df_normalization_preserves_integer_and_string_candidates():
    source = RankProfile(ballots=[RankBallot(ranking=[{1}, {"A"}])]).to_df().copy()
    source["Ranking_2"] = pd.Series([frozenset({" 1 "})])

    with pytest.warns(UserWarning, match="both str and int"):
        result = RankProfile(df=source, candidates=[1, " 1 "], max_ranking_length=2)

    assert result.candidates == result._matrix.candidates == (1, "1")
    with pytest.warns(UserWarning, match="both str and int"):
        assert result.ballots[0].ranking == (frozenset({1}), frozenset({"1"}))


@pytest.mark.parametrize("cell", [" A ", (" A ",)])
def test_from_df_normalization_rejects_non_set_ranking_cells(cell):
    source = RankProfile(ballots=[RankBallot(ranking=[{"A"}])]).to_df().copy()
    source["Ranking_1"] = pd.Series([cell], dtype=object)

    with pytest.raises(TypeError, match="Ranking cells must contain sets"):
        RankProfile(df=source, candidates=["A"], max_ranking_length=1)


@pytest.mark.parametrize("extra_column", ["Ranking_0", "Ranking_2", "Ranking_01", "Ranking_bad", 7])
def test_from_df_rejects_columns_that_would_be_omitted_from_ballots(extra_column):
    source = RankProfile(ballots=[RankBallot(ranking=[{"A"}])]).to_df()
    source[extra_column] = pd.Series([frozenset({"B"})])

    with pytest.raises(ProfileError, match="Unexpected columns"):
        RankProfile(df=source, max_ranking_length=1)


def test_from_df_rejects_duplicate_ranking_columns():
    source = RankProfile(ballots=[RankBallot(ranking=[{"A"}])]).to_df()
    source = pd.concat([source, source[["Ranking_1"]]], axis=1)

    with pytest.raises(ProfileError, match="columns must be unique"):
        RankProfile(df=source, max_ranking_length=1)


def test_from_df_accepts_reordered_ranking_columns():
    ballot = RankBallot(ranking=[{"A"}, {"B"}])
    source = RankProfile(ballots=[ballot]).to_df()
    source = source[["Weight", "Ranking_2", "Voter Set", "Ranking_1"]]

    result = RankProfile(df=source, max_ranking_length=2)

    assert result.ballots == (ballot,)
    assert result._matrix.ranking_columns == ("Ranking_1", "Ranking_2")


@pytest.mark.parametrize("invalid_candidate", [True, 1.0])
def test_from_df_checks_candidate_types_across_equal_ranking_sets(invalid_candidate):
    source = RankProfile(ballots=[RankBallot(ranking=[{1}])] * 2).to_df()
    source["Ranking_1"] = pd.Series([frozenset({1}), frozenset({invalid_candidate})])

    with pytest.raises(TypeError, match="candidate"):
        RankProfile(df=source, max_ranking_length=1)


def test_normalization_processes_candidate_names_once_per_distinct_set(monkeypatch):
    import votekit.pref_profile.pref_profile as profile_module

    original_normalize = profile_module._normalize_candidate_name
    visited = []

    def record_normalization(candidate):
        visited.append(candidate)
        return original_normalize(candidate)

    monkeypatch.setattr(profile_module, "_normalize_candidate_name", record_normalization)
    source = pd.DataFrame(
        {"Ranking_1": [frozenset({" A "}) for _ in range(1000)] + [frozenset({"A"})]}
    )
    normalized = profile_module._normalize_rank_df_candidates(source)

    assert visited == [" A ", "A"]
    assert normalized["Ranking_1"].tolist() == [frozenset({"A"})] * len(source)
    assert source["Ranking_1"].iloc[0] == frozenset({" A "})


@pytest.mark.parametrize("invalid_candidate", [True, 1.0])
@pytest.mark.parametrize("reverse", [False, True])
def test_normalization_does_not_hide_equal_invalid_candidate_types(invalid_candidate, reverse):
    from votekit.pref_profile.pref_profile import _normalize_rank_df_candidates

    cells = [frozenset({1}), frozenset({invalid_candidate})]
    if reverse:
        cells.reverse()
    with pytest.raises(TypeError, match="candidate"):
        _normalize_rank_df_candidates(pd.DataFrame({"Ranking_1": cells}))
