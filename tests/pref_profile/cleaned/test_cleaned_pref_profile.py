from votekit.pref_profile import PreferenceProfile, CleanedProfile

# from votekit.ballot import Ballot
import pandas as pd


def test_init():
    empty_profile = CleanedProfile(
        parent_profile=PreferenceProfile(), df_index_column=[]
    )

    assert empty_profile.ballots == ()
    assert not empty_profile.candidates
    assert empty_profile.df.equals(
        pd.DataFrame(
            {
                "Voter Set": [],
                "Weight": [],
            }
        )
    )
    assert not empty_profile.candidates_cast
    assert not empty_profile.total_ballot_wt
    assert not empty_profile.num_ballots
    assert empty_profile.max_ranking_length == 0

    assert empty_profile.parent_profile == PreferenceProfile()
    assert empty_profile.df_index_column == []
    assert empty_profile.no_weight_altr_ballot_indices == set()
    assert empty_profile.no_ranking_and_no_scores_altr_ballot_indices == set()
    assert empty_profile.valid_but_altr_ballot_indices == set()
    assert empty_profile.unaltr_ballot_indices == set()


# def test_parents():
#     profile = PreferenceProfile(
#         ballots=[
#             Ballot(ranking=[{"A"}, {"B"}], weight=1),
#             Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=1),
#             Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=3),
#             Ballot(ranking=({"A"},)),
#             Ballot(ranking=({"B"},), weight=0),
#         ]
#     )

#     clean_1 = CleanedProfile(parent_profile=profile, no_weight_altr_ballot_indices=[4])
#     clean_2 = CleanedProfile(
#         parent_profile=clean_1, empty_ranking_and_no_scores_altr_ballot_indices=[1]
#     )

#     assert clean_2.parent_profile == clean_1
#     assert clean_1.parent_profile == profile
#     assert clean_2.parent_profile.parent_profile == profile

# test errors for index sets
# test reindexing of df

# test group ballots wrning
# test str
