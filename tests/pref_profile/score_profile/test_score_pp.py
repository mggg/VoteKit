from votekit.pref_profile import ScoreProfile


def test_init():
    empty_profile = ScoreProfile()
    print(type(empty_profile))
    assert empty_profile.ballots == tuple()
    assert not empty_profile.candidates
    assert not empty_profile.candidates_cast
    assert not empty_profile.total_ballot_wt
    assert not empty_profile.num_ballots


# def test_unique_cands_validator():
#     with pytest.raises(ValueError, match="All candidates must be unique."):
#         ScoreProfile(candidates=("A", "A", "B"))

#     ScoreProfile(candidates=("A", "B"))


# def test_strip_whitespace():
#     pp = ScoreProfile(candidates=("A ", " B", " C "))

#     assert pp.candidates == ("A", "B", "C")


# def test_ballots_frozen():
#     p = ScoreProfile(ballots=[ScoreBallot()])
#     b_list = p.ballots

#     assert b_list == (ScoreBallot(),)

#     with pytest.raises(
#         AttributeError,
#         match="Cannot modify frozen instance: tried to set 'ballots'",
#     ):
#         p.ballots = (ScoreBallot(weight=5),)


# def test_candidates_frozen():
#     profile_no_cands = ScoreProfile(
#         ballots=[
#             ScoreBallot(scores={"A": 4}),
#             ScoreBallot(scores={"B": 4}),
#             ScoreBallot(scores={"C": 4}),
#         ]
#     )
#     assert set(profile_no_cands.candidates) == set(["A", "B", "C"])
#     assert set(profile_no_cands.candidates_cast) == set(["A", "B", "C"])

#     with pytest.raises(
#         AttributeError, match="Cannot modify frozen instance: tried to set 'candidates'"
#     ):
#         profile_no_cands.candidates = tuple()

#     with pytest.raises(
#         AttributeError,
#         match="Cannot modify frozen instance: tried to set 'candidates_cast'",
#     ):
#         profile_no_cands.candidates_cast = tuple()


# def test_get_candidates_received_votes():
#     profile_w_cands = ScoreProfile(
#         ballots=[
#             ScoreBallot(scores={"A": 4}),
#             ScoreBallot(scores={"B": 4}),
#             ScoreBallot(scores={"C": 4}),
#         ],
#         candidates=("A", "B", "C", "D", "E"),
#     )
#     vote_cands = profile_w_cands.candidates_cast
#     all_cands = profile_w_cands.candidates

#     assert set(all_cands) == {"A", "B", "C", "D", "E"}
#     assert set(vote_cands) == {
#         "A",
#         "B",
#         "C",
#     }
