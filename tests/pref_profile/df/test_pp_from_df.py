from votekit.pref_profile import PreferenceProfile
from votekit.ballot import Ballot
import pytest
import pandas as pd

profile = PreferenceProfile(
    ballots=(
        Ballot(ranking=[{"A"}, {"B"}]),
        Ballot(ranking=[{"C"}, {"B"}]),
        Ballot(scores={"A": 4, "E": 4}),
        Ballot(scores={"A": 4, "E": 4}, ranking=[{"A"}, {"B"}]),
        Ballot(ranking=[{"D"}], weight=0),
    ),
    candidates=("A", "B", "C", "D", "E"),
    max_ranking_length=3,
)
df = profile.df


def test_from_df():
    new_profile = PreferenceProfile(
        df=df,
        candidates=profile.candidates,
        max_ranking_length=profile.max_ranking_length,
        contains_rankings=True,
        contains_scores=True,
    )

    assert new_profile == profile
    assert set(new_profile.candidates_cast) == {"A", "B", "C", "E"}


def test_from_df_init_errors():
    with pytest.raises(
        ValueError,
        match="When providing a dataframe and no ballot list to the init method, one of "
        "contains_rankings and contains_scores must be True.",
    ):
        PreferenceProfile(df=df)

    with pytest.raises(
        ValueError,
        match="When providing a dataframe and no ballot list to the init method, if "
        "contains_rankings is True, max_ranking_length must be provided and be non-zero.",
    ):
        PreferenceProfile(df=df, contains_rankings=True)

    with pytest.raises(
        ValueError,
        match="When providing a dataframe and no ballot list to the init method, if "
        "contains_rankings is True, max_ranking_length must be provided and be non-zero.",
    ):
        PreferenceProfile(df=df, contains_rankings=True, max_ranking_length=0)

    with pytest.raises(
        ValueError,
        match="When providing a dataframe and no ballot list to the init method, "
        "candidates must be provided.",
    ):
        PreferenceProfile(df=df, contains_rankings=True, max_ranking_length=3)


def test_from_df_validation_errors():
    with pytest.raises(ValueError, match="Weight column not in dataframe:"):
        PreferenceProfile(
            df=pd.DataFrame(columns=["Voter Set"]),
            contains_rankings=True,
            max_ranking_length=3,
            candidates=["A"],
        )

    with pytest.raises(ValueError, match="Voter Set column not in dataframe:"):
        PreferenceProfile(
            df=pd.DataFrame(columns=["Weight"]),
            contains_rankings=True,
            max_ranking_length=3,
            candidates=["A"],
        )

    with pytest.raises(ValueError, match="Index not named 'Ballot Index':"):
        PreferenceProfile(
            df=pd.DataFrame(columns=["Weight", "Voter Set"]),
            contains_rankings=True,
            max_ranking_length=3,
            candidates=["A"],
        )

    with pytest.raises(
        ValueError, match="Ranking column 'Ranking_2' not in dataframe:"
    ):
        df = pd.DataFrame(columns=["Weight", "Voter Set", "Ranking_1", "Ranking_3"])
        df.index.name = "Ballot Index"
        PreferenceProfile(
            df=df,
            contains_rankings=True,
            max_ranking_length=3,
            candidates=["A"],
        )

    with pytest.raises(ValueError, match="Candidate column 'B' not in dataframe:"):
        df = pd.DataFrame(columns=["Weight", "Voter Set", "A", "C"])
        df.index.name = "Ballot Index"
        PreferenceProfile(
            df=df,
            contains_scores=True,
            candidates=["A", "B", "C"],
        )
