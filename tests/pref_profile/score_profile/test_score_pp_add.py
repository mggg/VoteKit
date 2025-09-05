from votekit.ballot import ScoreBallot
from votekit.pref_profile import ScoreProfile


def test_add_profiles():
    profile_1 = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2, "B": 2}, weight=2),
            ScoreBallot(scores={"A": 2, "C": 2}, voter_set={"Chris"}),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
    )

    profile_2 = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"D": 2, "E": 2}, weight=2),
            ScoreBallot(scores={"D": 2, "E": 2, "F": 3.1}, weight=2),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ],
        candidates=["D", "E", "F"],
    )
    summed_profile = profile_1 + profile_2
    true_summed_profile = ScoreProfile(
        ballots=(
            ScoreBallot(scores={"A": 2, "B": 2}, weight=2),
            ScoreBallot(scores={"A": 2, "C": 2}, voter_set={"Chris"}),
            ScoreBallot(),
            ScoreBallot(weight=0),
            ScoreBallot(scores={"D": 2, "E": 2}, weight=2),
            ScoreBallot(scores={"D": 2, "E": 2, "F": 3.1}, weight=2),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ),
        candidates=["A", "B", "C", "D", "E", "F"],
    )

    assert set(summed_profile.candidates) == set(["A", "B", "C", "D", "E", "F"])
    assert isinstance(summed_profile, ScoreProfile)
    assert true_summed_profile == summed_profile
