from votekit.pref_profile import PreferenceProfile, CleanedRankProfile
from votekit.ballot import Ballot
from votekit.cleaning import remove_cand_ranked_profile

profile_no_ties = PreferenceProfile(
    ballots=[
        Ballot(ranking=[{"A"}, {"B"}], weight=1),
        Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=1 / 2),
        Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=3),
    ]
)

profile_with_ties = PreferenceProfile(
    ballots=[
        Ballot(ranking=[{"A", "B"}], weight=1),
        Ballot(ranking=[{"A", "B", "C"}], weight=1 / 2),
        Ballot(ranking=[{"A"}, {"C"}, {"B"}], weight=3),
    ]
)


def test_remove_cand():
    cleaned_profile = remove_cand_ranked_profile("A", profile_no_ties)

    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile_no_ties
    assert cleaned_profile.ballots == (
        Ballot(ranking=[frozenset(), {"B"}], weight=1),
        Ballot(ranking=[frozenset(), {"B"}, {"C"}], weight=1 / 2),
        Ballot(ranking=[{"C"}, {"B"}, frozenset()], weight=3),
    )
    assert cleaned_profile != profile_no_ties
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == set()
    assert cleaned_profile.nonempty_altr_idxs == {0, 1, 2}
    assert cleaned_profile.unaltr_idxs == set()


def test_remove_mult_cands():
    cleaned_profile = remove_cand_ranked_profile(["A", "B"], profile_no_ties)

    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile_no_ties

    assert set(cleaned_profile.group_ballots().ballots) == set(
        [
            Ballot(ranking=[frozenset(), frozenset()], weight=1),
            Ballot(ranking=[frozenset(), frozenset(), {"C"}], weight=1 / 2),
            Ballot(ranking=[{"C"}, frozenset(), frozenset()], weight=3),
        ]
    )
    assert cleaned_profile != profile_no_ties
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == set()
    assert cleaned_profile.nonempty_altr_idxs == {0, 1, 2}
    assert cleaned_profile.unaltr_idxs == set()


def test_remove_cand_with_ties():

    cleaned_profile = remove_cand_ranked_profile(["A", "B"], profile_with_ties)
    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile_with_ties

    assert set(cleaned_profile.group_ballots().ballots) == set(
        [
            Ballot(ranking=[frozenset()], weight=1),
            Ballot(ranking=[{"C"}], weight=1 / 2),
            Ballot(ranking=[frozenset(), {"C"}, frozenset()], weight=3),
        ]
    )
    assert cleaned_profile != profile_with_ties
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == set()
    assert cleaned_profile.nonempty_altr_idxs == {0, 1, 2}
    assert cleaned_profile.unaltr_idxs == set()
