from votekit.pref_profile import PreferenceProfile, CleanedRankProfile
from votekit.ballot import Ballot
from votekit.cleaning import remove_repeat_cands_ranked_profile


def test_remove_repeated_candidates():
    ballot = Ballot(ranking=[{"A"}, {"A"}, {"B"}, {"C"}], weight=1)
    ballot_tuple = (ballot, ballot)
    profile = PreferenceProfile(ballots=ballot_tuple)
    cleaned_profile = remove_repeat_cands_ranked_profile(profile)

    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile

    assert cleaned_profile.group_ballots().ballots == (
        Ballot(ranking=({"A"}, frozenset(), {"B"}, {"C"}), weight=2),
    )

    assert cleaned_profile != profile
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == set()
    assert cleaned_profile.nonempty_altr_idxs == {0, 1}
    assert cleaned_profile.unaltr_idxs == set()


def test_remove_repeated_candidates_ties():
    profile = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=[{"C", "A"}, {"A", "C"}, {"B"}],
            ),
            Ballot(
                ranking=[{"A", "C"}, {"C", "A"}, {"B"}],
            ),
        ]
    )
    cleaned_profile = remove_repeat_cands_ranked_profile(profile)

    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile

    assert cleaned_profile.group_ballots().ballots == (
        Ballot(ranking=[{"C", "A"}, frozenset(), {"B"}], weight=2),
    )

    assert cleaned_profile != profile
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == set()
    assert cleaned_profile.nonempty_altr_idxs == {0, 1}
    assert cleaned_profile.unaltr_idxs == set()
