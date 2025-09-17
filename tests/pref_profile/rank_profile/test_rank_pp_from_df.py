from votekit.pref_profile import RankProfile, ProfileError
from votekit.ballot import RankBallot, ScoreBallot
import pytest
import pandas as pd

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
        match="Profile cannot contain RankBallots and ScoreBallots. "
        "There are 1 ScoreBallots and 1 RankBallots.",
    ):
        RankProfile(
            ballots=(RankBallot(ranking=[{"Chris"}]), ScoreBallot(scores={"Chris": 1}))
        )


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

    with pytest.raises(
        ProfileError, match="Ranking column 'Ranking_2' not in dataframe:"
    ):
        df = pd.DataFrame(columns=["Weight", "Voter Set", "Ranking_1", "Ranking_3"])
        df.index.name = "Ballot Index"
        RankProfile(
            df=df,
            max_ranking_length=3,
            candidates=["A"],
        )
