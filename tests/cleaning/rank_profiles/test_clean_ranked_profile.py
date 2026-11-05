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
            RankBallot(ranking=(), weight=2),
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
    assert adj_profile.no_rank_altr_idxs == {3}
    assert adj_profile.nonempty_altr_idxs == {0, 1, 2}
    assert adj_profile.unaltr_idxs == {4, 5}
    assert adj_profile.candidates == profile.candidates
    assert adj_profile.max_ranking_length == profile.max_ranking_length


def test_clean_profile_change_defaults(profile):
    adj_profile = clean_rank_profile(
        profile,
        remove_A_from_ballot,
        remove_null_ballots=False,
        remove_zero_weight_ballots=False,
        retain_original_candidate_list=False,
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
            RankBallot(weight=2),
        )
    )

    assert adj_profile.candidates != profile.candidates
    assert adj_profile.max_ranking_length == profile.max_ranking_length
    assert adj_profile.no_wt_altr_idxs == set()
    assert adj_profile.no_rank_altr_idxs == {3}
    assert adj_profile.nonempty_altr_idxs == {0, 1, 2}
    assert adj_profile.unaltr_idxs == {4, 5}
    assert set(adj_profile.candidates) == {"B", "C"}
    assert adj_profile.max_ranking_length == profile.max_ranking_length


def replace_candidate(old_cand, new_cand):
    def clean_ranking(ranking):
        return tuple(
            frozenset(
                new_cand if candidate == old_cand else candidate for candidate in candidate_set
            )
            if isinstance(candidate_set, frozenset)
            else candidate_set
            for candidate_set in ranking
        )

    return clean_ranking


@pytest.mark.parametrize(
    "retain_original_candidate_list, expected_candidates",
    [(True, ("A", "B")), (False, ("B",))],
)
def test_clean_profile_inserts_unranked_declared_candidate(
    retain_original_candidate_list, expected_candidates
):
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}], weight=1)],
        candidates=("A", "B"),
    )
    assert profile.candidates_cast == ("A",)

    adj_profile = clean_rank_profile(
        profile,
        replace_candidate("A", "B"),
        retain_original_candidate_list=retain_original_candidate_list,
    )

    assert adj_profile.ballots == (RankBallot(ranking=[{"B"}], weight=1),)
    assert adj_profile.candidates == expected_candidates
    assert adj_profile.candidates_cast == ("B",)


@pytest.mark.parametrize("retain_original_candidate_list", [True, False])
def test_clean_profile_insert_undeclared_candidate_on_positive_weight_ballot_errors(
    retain_original_candidate_list,
):
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}], weight=1)],
        candidates=("A", "B"),
    )

    with pytest.raises(
        ValueError,
        match="Rankings on positive-weight ballots contain candidates outside the declared"
        " candidates.",
    ):
        clean_rank_profile(
            profile,
            replace_candidate("A", "C"),
            retain_original_candidate_list=retain_original_candidate_list,
        )


def test_clean_profile_strips_whitespace_to_existing_candidate_set():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=1)],
        candidates=("A", "B"),
    )

    adj_profile = clean_rank_profile(profile, replace_candidate("A", " A "))

    assert adj_profile.ballots == profile.ballots
    assert adj_profile.candidates_cast == ("A", "B")
    assert adj_profile.unaltr_idxs == {0}
    assert adj_profile.nonempty_altr_idxs == set()


def test_clean_profile_boolean_candidate_errors():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=1)],
        candidates=("A", "B"),
    )

    with pytest.raises(TypeError, match=r"Boolean candidate\(s\) found in cleaned rankings"):
        clean_rank_profile(profile, replace_candidate("A", True))


def test_clean_profile_stripped_name_collision_warns():
    # "1" stays undeclared so the collision warning comes from cleaning, not profile construction.
    # Undeclared candidates are only allowed on zero-weight ballots.
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {1}], weight=1),
            RankBallot(ranking=[{"B"}], weight=0),
        ],
        candidates=("A", 1, "B"),
    )

    with pytest.warns(UserWarning, match="will be treated as separate candidates"):
        adj_profile = clean_rank_profile(
            profile, replace_candidate("B", "1"), remove_zero_weight_ballots=False
        )

    assert adj_profile.ballots[1] == RankBallot(ranking=[{"1"}], weight=0)
    assert adj_profile.candidates_cast == (1, "A")


def tie_A_with_B(ranking):
    return tuple(
        candidate_set | {"B"} if candidate_set == frozenset({"A"}) else candidate_set
        for candidate_set in ranking
    )


def test_clean_profile_tie_exceeding_max_ranking_length_errors():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}], weight=1)],
        candidates=("A", "B"),
        max_ranking_length=1,
    )

    with pytest.raises(
        ValueError, match="ranks 2 distinct candidates, exceeding max_ranking_length"
    ):
        clean_rank_profile(profile, tie_A_with_B)


def test_clean_profile_tie_on_removed_ballot_is_allowed():
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=[{"B"}], weight=1),
            RankBallot(ranking=[{"A"}], weight=0),
        ],
        candidates=("A", "B"),
        max_ranking_length=1,
    )

    adj_profile = clean_rank_profile(profile, tie_A_with_B)

    assert adj_profile.ballots == (RankBallot(ranking=[{"B"}], weight=1),)
    assert adj_profile.max_ranking_length == 1
