# from unnamed_rcv_thing.profile import PreferenceProfile
# from unnamed_rcv_thing.ballot import Ballot


# def test_unique_cands():
#     profile = PreferenceProfile(
#         ballots=[
#             Ballot(ranking=["A", "B", "C"], weight=1),
#             Ballot(ranking=["B", "C", "E"], weight=1),
#         ]
#     )

#     cands = profile.get_candidates()
#     assert {"A", "B", "C", "E"} == set(cands)
