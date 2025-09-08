from votekit.ballot import ScoreBallot, RankBallot
from votekit.pref_profile import RankProfile, ScoreProfile
from votekit.pref_profile.utils import (
    rank_profile_to_ballot_dict,
    score_profile_to_ballot_dict,
)


def test_rank_to_ballot_dict():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"})),
            RankBallot(ranking=({"A"}, {"B"}), weight=3 / 2),
            RankBallot(ranking=({"C"}, {"B"}), weight=2),
        )
    )
    rv = rank_profile_to_ballot_dict(profile, standardize=False)
    assert rv[RankBallot(ranking=({"A"}, {"B"}))] == 5 / 2
    assert rv[RankBallot(ranking=({"C"}, {"B"}))] == 2 / 1

    rv = rank_profile_to_ballot_dict(profile, standardize=True)
    assert rv[RankBallot(ranking=({"A"}, {"B"}))] == 5 / 9
    assert rv[RankBallot(ranking=({"C"}, {"B"}))] == 4 / 9


def test_score_to_ballot_dict():
    profile = ScoreProfile(
        ballots=(
            ScoreBallot(scores={"A": 4}, weight=2),
            ScoreBallot(scores={"A": 3}),
        )
    )
    rv = score_profile_to_ballot_dict(profile, standardize=False)
    assert rv[ScoreBallot(scores={"A": 4})] == 2 / 1
    assert rv[ScoreBallot(scores={"A": 3})] == 1

    rv = score_profile_to_ballot_dict(profile, standardize=True)
    assert rv[ScoreBallot(scores={"A": 4})] == 2 / 3
    assert rv[ScoreBallot(scores={"A": 3})] == 1 / 3
