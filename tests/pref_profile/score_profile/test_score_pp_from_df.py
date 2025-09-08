from votekit.pref_profile import ScoreProfile, ProfileError
from votekit.ballot import ScoreBallot, RankBallot
import pytest
import pandas as pd

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
        match="Profile cannot contain RankBallots and ScoreBallots. ScoreBallots"
        r" appear at indices \[1\], RankBallots appear at indices"
        r" \[0\].",
    ):
        ScoreProfile(
            ballots=(RankBallot(ranking=[{"Chris"}]), ScoreBallot(scores={"Chris": 1}))
        )

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
