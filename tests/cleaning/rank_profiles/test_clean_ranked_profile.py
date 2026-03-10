from votekit.ballot import RankBallot
from votekit.cleaning import clean_rank_profile
from votekit.pref_profile import CleanedRankProfile, RankProfile

profile = RankProfile(
    ballots=[
        RankBallot(ranking=[{"A"}, {"B"}], weight=1),
        RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=1),
        RankBallot(ranking=[{"C"}, {"B"}, {"A"}], weight=3),
        RankBallot(ranking=({"A"},)),
        RankBallot(ranking=({"B"},), weight=0),
    ]
)


def test_clean_profile_with_defaults():
    adj_profile = clean_rank_profile(
        profile,
        lambda x: tuple(c_set - {"A"} if isinstance(c_set, frozenset) else c_set for c_set in x),
    )

    assert isinstance(adj_profile, CleanedRankProfile)
    assert adj_profile.parent_profile == profile
    assert adj_profile.ballots == (
        RankBallot(ranking=[frozenset(), {"B"}], weight=1),
        RankBallot(ranking=[frozenset(), {"B"}, {"C"}], weight=1),
        RankBallot(ranking=[{"C"}, {"B"}, frozenset()], weight=3),
        RankBallot(ranking=[frozenset()]),
    )
    assert adj_profile != profile

    assert adj_profile.no_wt_altr_idxs == set()
    assert adj_profile.no_rank_altr_idxs == set()
    assert adj_profile.nonempty_altr_idxs == {0, 1, 2, 3}
    assert adj_profile.unaltr_idxs == {4}


def test_clean_profile_change_defaults():
    adj_profile = clean_rank_profile(
        profile,
        lambda x: tuple(c_set - {"A"} if isinstance(c_set, frozenset) else c_set for c_set in x),
        remove_empty_ballots=False,
        remove_zero_weight_ballots=False,
        retain_original_candidate_list=True,
    )
    assert isinstance(adj_profile, CleanedRankProfile)
    assert adj_profile.parent_profile == profile
    assert set(adj_profile.ballots) == set(
        (
            RankBallot(ranking=[frozenset(), {"B"}], weight=1),
            RankBallot(ranking=[frozenset(), {"B"}, {"C"}], weight=1),
            RankBallot(
                ranking=[
                    {"C"},
                    {"B"},
                    frozenset(),
                ],
                weight=3,
            ),
            RankBallot(ranking=(frozenset(),)),
            RankBallot(ranking=({"B"},), weight=0),
        )
    )

    assert adj_profile.candidates == profile.candidates
    assert adj_profile.max_ranking_length == 3
    assert adj_profile.no_wt_altr_idxs == set()
    assert adj_profile.no_rank_altr_idxs == set()
    assert adj_profile.nonempty_altr_idxs == {0, 1, 2, 3}
    assert adj_profile.unaltr_idxs == {4}
