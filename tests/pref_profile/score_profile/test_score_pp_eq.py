from votekit.ballot import ScoreBallot
from votekit.pref_profile import ScoreProfile


def test_profile_equals_scores():
    profile1 = ScoreProfile(
        ballots=(
            ScoreBallot(scores={"D": 2, "E": 2}, weight=2),
            ScoreBallot(scores={"D": 2, "E": 2}, weight=2),
            ScoreBallot(),
            ScoreBallot(weight=0),
        )
    )
    profile2 = ScoreProfile(
        ballots=(
            ScoreBallot(scores={"D": 2, "E": 2}, weight=4),
            ScoreBallot(),
            ScoreBallot(weight=0),
        )
    )
    assert profile1 == profile2


def test_profile_not_equals_candidates():
    profile1 = ScoreProfile(
        ballots=(
            ScoreBallot(scores={"D": 2, "E": 2}, weight=4),
            ScoreBallot(),
            ScoreBallot(weight=0),
        )
    )
    profile2 = ScoreProfile(
        ballots=(
            ScoreBallot(scores={"D": 2, "E": 2}, weight=4),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ),
        candidates=["A", "B", "C", "D", "E"],
    )

    assert profile1 != profile2


def test_profile_not_equals_cand_cast():
    profile1 = ScoreProfile(
        ballots=(
            ScoreBallot(scores={"D": 2, "E": 2, "F": 1}, weight=4),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ),
        candidates=["D", "E", "F"],
    )
    profile2 = ScoreProfile(
        ballots=(
            ScoreBallot(scores={"D": 2, "E": 2}, weight=4),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ),
        candidates=["D", "E", "F"],
    )

    assert profile1 != profile2


def test_profile_not_equals_ballot_wt():
    profile1 = ScoreProfile(
        ballots=(
            ScoreBallot(scores={"D": 2, "E": 2, "F": 1}, weight=4),
            ScoreBallot(scores={"D": 2, "E": 2, "F": 1}, weight=4),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ),
        candidates=["D", "E", "F"],
    )
    profile2 = ScoreProfile(
        ballots=(
            ScoreBallot(scores={"D": 2, "E": 2, "F": 1}, weight=2),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ),
        candidates=["D", "E", "F"],
    )

    assert profile1 != profile2
