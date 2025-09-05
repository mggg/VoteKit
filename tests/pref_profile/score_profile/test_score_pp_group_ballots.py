from votekit.ballot import ScoreBallot
from votekit.pref_profile import ScoreProfile


def test_pp_group_ballots_scores():
    profile = ScoreProfile(
        ballots=(
            ScoreBallot(scores={"D": 2, "E": 2}, weight=2, voter_set={"Chris"}),
            ScoreBallot(scores={"D": 2, "E": 2}, weight=2, voter_set={"Moon", "Peter"}),
            ScoreBallot(
                scores={"D": 2, "E": 2},
                weight=2,
            ),
            ScoreBallot(),
        )
    )

    pp = profile.group_ballots()
    assert set(pp.ballots) == set(
        (
            ScoreBallot(
                scores={"D": 2, "E": 2},
                weight=6,
                voter_set={"Chris", "Moon", "Peter"},
            ),
            ScoreBallot(),
        )
    )
    assert set(pp.candidates) == set(profile.candidates)
    assert profile == pp
