from votekit.pref_profile import PreferenceProfile, CleanedProfile
from votekit.ballot import Ballot
from votekit.cleaning import clean_profile

profile = PreferenceProfile(
    ballots=[
        Ballot(ranking=[{"A"}, {"B"}], weight=1),
        Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=1),
        Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=3),
        Ballot(ranking=({"A"},)),
        Ballot(ranking=({"B"},), weight=0),
    ]
)


def test_clean_profile_with_defaults():
    adj_profile = clean_profile(
        profile,
        lambda x: Ballot(
            ranking=[c_set for c_set in x.ranking if "A" not in c_set], weight=x.weight
        ),
    )

    assert isinstance(adj_profile, CleanedProfile)
    assert adj_profile.parent_profile == profile
    assert adj_profile.ballots == (
        Ballot(ranking=[{"B"}], weight=1),
        Ballot(ranking=[{"B"}, {"C"}], weight=1),
        Ballot(ranking=[{"C"}, {"B"}], weight=3),
    )
    assert adj_profile != profile

    assert adj_profile.no_weight_altr_ballot_indices == []
    assert adj_profile.empty_ranking_and_no_scores_altr_ballot_indices == [3]
    assert adj_profile.nonempty_altr_ballot_indices == [0, 1, 2]
    assert adj_profile.unaltr_ballot_indices == [4]


def test_clean_profile_change_defaults():
    adj_profile = clean_profile(
        profile,
        lambda x: Ballot(
            ranking=[c_set for c_set in x.ranking if "A" not in c_set], weight=x.weight
        ),
        remove_empty_ballots=False,
        remove_zero_weight_ballots=False,
        retain_original_candidate_list=True,
        retain_original_max_ballot_length=False,
    )

    assert isinstance(adj_profile, CleanedProfile)
    assert adj_profile.parent_profile == profile
    assert set(adj_profile.ballots) == set(
        (
            Ballot(ranking=[{"B"}], weight=1),
            Ballot(ranking=[{"B"}, {"C"}], weight=1),
            Ballot(ranking=[{"C"}, {"B"}], weight=3),
            Ballot(ranking=tuple()),
            Ballot(ranking=({"B"},), weight=0),
        )
    )
    assert adj_profile.candidates == profile.candidates
    assert adj_profile.max_ballot_length == 2

    assert adj_profile.no_weight_altr_ballot_indices == []
    assert adj_profile.empty_ranking_and_no_scores_altr_ballot_indices == [3]
    assert adj_profile.nonempty_altr_ballot_indices == [0, 1, 2]
    assert adj_profile.unaltr_ballot_indices == [4]
