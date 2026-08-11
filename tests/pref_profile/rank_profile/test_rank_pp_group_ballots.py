import warnings

from pandas.errors import PerformanceWarning

from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile


def test_pp_group_ballots_ranking():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=1, voter_set={"Chris"}),
            RankBallot(
                ranking=({"A"}, {"B"}, {"C"}),
                weight=2,
                voter_set={"Moon", "Peter"},
            ),
        ),
        candidates=("A", "B", "C", "D"),
    )

    pp = profile.group_ballots()
    assert pp.ballots == (
        RankBallot(
            ranking=({"A"}, {"B"}, {"C"}),
            weight=5,
            voter_set={"Chris", "Moon", "Peter"},
        ),
    )
    assert set(pp.candidates) == set(profile.candidates)
    assert profile == pp


def _wide_rank_profile(n_cands: int, n_ballots: int) -> RankProfile:
    candidates = [f"C{i}" for i in range(n_cands)]
    ballots = tuple(
        RankBallot(
            ranking=tuple({c} for c in candidates[i % n_cands :] + candidates[: i % n_cands]),
            weight=1,
        )
        for i in range(n_ballots)
    )
    return RankProfile(ballots=ballots, candidates=tuple(candidates))


def test_pp_group_ballots_ranking_no_fragmentation_warning():
    profile = _wide_rank_profile(120, 300)

    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        profile.group_ballots()


def test_pp_group_ballots_ranking_wide_profile_groups_correctly():
    profile = _wide_rank_profile(120, 300)

    pp = profile.group_ballots()
    assert pp.num_ballots == 120
    assert pp.total_ballot_wt == profile.total_ballot_wt
    assert set(pp.df.columns) == set(profile.df.columns)
    assert {b.ranking for b in pp.ballots} == {b.ranking for b in profile.ballots}

    # the i-th ballot ranks rotation i % 120, so 300 ballots give 60 rotations seen 3
    # times and 60 seen twice
    weights = sorted(b.weight for b in pp.ballots)
    assert weights == [2] * 60 + [3] * 60
    assert profile == pp


def test_pp_group_ballots_ranking_no_ranking_ballots():
    profile = RankProfile(
        ballots=(RankBallot(weight=1),) * 2,
        candidates=tuple(f"C{i}" for i in range(120)),
        max_ranking_length=120,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        pp = profile.group_ballots()

    assert pp.num_ballots == 1
    assert pp.total_ballot_wt == 2
