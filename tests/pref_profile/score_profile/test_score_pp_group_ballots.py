import warnings

from pandas.errors import PerformanceWarning

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


def _wide_score_profile(n_cands: int, n_ballots: int) -> ScoreProfile:
    candidates = tuple(f"C{i}" for i in range(n_cands))
    ballots = tuple(
        ScoreBallot(scores={c: (i % 5) + 1 for c in candidates}, weight=1) for i in range(n_ballots)
    )
    return ScoreProfile(ballots=ballots, candidates=candidates)


def test_pp_group_ballots_scores_no_fragmentation_warning():
    profile = _wide_score_profile(120, 300)

    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        profile.group_ballots()


def test_pp_group_ballots_scores_wide_profile_groups_correctly():
    profile = _wide_score_profile(120, 300)

    pp = profile.group_ballots()
    assert pp.num_ballots == 5
    assert pp.total_ballot_wt == profile.total_ballot_wt
    assert set(pp.ballots) == set(
        ScoreBallot(scores={c: (i % 5) + 1 for c in profile.candidates}, weight=60)
        for i in range(5)
    )
    assert set(pp.df.columns) == set(profile.df.columns)
    assert profile == pp
