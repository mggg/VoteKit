import pytest

from votekit.ballot import RankBallot
from votekit.cleaning import clean_rank_profile
from votekit.pref_profile import CleanedRankProfile, RankProfile


@pytest.fixture
def profile():
    return RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}], weight=1),
            RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=1),
            RankBallot(ranking=[{"C"}, {"B"}, {"A"}], weight=3),
            RankBallot(ranking=({"A"},)),
            RankBallot(ranking=({"B"},), weight=0),
        ]
    )


@pytest.fixture
def profile_with_repeated_candidates():
    return RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}, {"C"}, {"B"}], weight=1),
            RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=1),
            RankBallot(ranking=[{"C"}, {"B"}, {"A"}], weight=3),
            RankBallot(ranking=({"A"},)),
            RankBallot(ranking=({"B"},), weight=0),
        ]
    )


@pytest.fixture
def profile_with_ties():
    return RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}, {"C", "D"}], weight=1),
            RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=1),
            RankBallot(ranking=[{"C"}, {"B"}, {"A"}], weight=3),
            RankBallot(ranking=({"A"},)),
            RankBallot(ranking=({"B"},), weight=0),
        ],
        max_ranking_length=4,
    )


def remove_A_from_ballot(ranking):
    return tuple(c_set - {"A"} if isinstance(c_set, frozenset) else c_set for c_set in ranking)


def condense_ranking(ranking):
    max_ranking_length = len(ranking)
    condensed_ranking = [cand_set for cand_set in ranking if cand_set]

    if len(condensed_ranking) < max_ranking_length:
        condensed_ranking += [frozenset("~")] * (max_ranking_length - len(condensed_ranking))

    return tuple(condensed_ranking)


def test_clean_profile_with_defaults(profile):
    adj_profile = clean_rank_profile(
        profile,
        remove_A_from_ballot,
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


def test_clean_profile_change_defaults(profile):
    adj_profile = clean_rank_profile(
        profile,
        remove_A_from_ballot,
        remove_empty_ballots=False,
        remove_zero_weight_ballots=False,
        retain_original_candidate_list=True,
        reduce_max_ranking_length=False,
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
    assert adj_profile.max_ranking_length == profile.max_ranking_length
    assert adj_profile.no_wt_altr_idxs == set()
    assert adj_profile.no_rank_altr_idxs == set()
    assert adj_profile.nonempty_altr_idxs == {0, 1, 2, 3}
    assert adj_profile.unaltr_idxs == {4}


def test_clean_profile_with_reduce_max_ranking_length(
    profile_with_repeated_candidates, profile_with_ties
):
    adj_profile = clean_rank_profile(
        profile_with_repeated_candidates,
        lambda b: condense_ranking(remove_A_from_ballot(b)),
        reduce_max_ranking_length=True,
    )

    assert adj_profile.ballots == (
        RankBallot(ranking=[{"B"}, {"C"}, ("B")], weight=1),
        RankBallot(ranking=[{"B"}, {"C"}], weight=1),
        RankBallot(ranking=[{"C"}, {"B"}], weight=3),
    )
    assert adj_profile != profile_with_repeated_candidates
    assert adj_profile.max_ranking_length != profile_with_repeated_candidates.max_ranking_length
    assert adj_profile.max_ranking_length == 3
    assert adj_profile.max_ranking_length > adj_profile.max_candidates_ranked

    adj_profile = clean_rank_profile(
        profile_with_ties,
        lambda b: condense_ranking(remove_A_from_ballot(b)),
        reduce_max_ranking_length=True,
    )

    assert adj_profile.ballots == (
        RankBallot(ranking=[{"B"}, {"C", "D"}], weight=1),
        RankBallot(ranking=[{"B"}, {"C"}], weight=1),
        RankBallot(ranking=[{"C"}, {"B"}], weight=3),
    )
    assert adj_profile != profile_with_ties
    assert adj_profile.max_ranking_length != profile_with_ties.max_ranking_length
    assert adj_profile.max_ranking_length == 3
    assert adj_profile.max_ranking_length == adj_profile.max_candidates_ranked
