from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
import pytest
import dataclasses


def test_init():
    empty_profile = PreferenceProfile()
    assert empty_profile.ballots == ()
    assert not empty_profile.candidates
    assert not empty_profile.candidates_cast
    assert not empty_profile.total_ballot_wt
    assert not empty_profile.num_ballots
    assert empty_profile.max_ranking_length == 0


def test_unique_cands_validator():
    with pytest.raises(ValueError, match="All candidates must be unique."):
        PreferenceProfile(candidates=("A", "A", "B"))

    PreferenceProfile(candidates=("A", "B"))


def test_strip_whitespace():
    pp = PreferenceProfile(candidates=("A ", " B", " C "))

    assert pp.candidates == ("A", "B", "C")


def test_ballots_frozen():
    p = PreferenceProfile(ballots=[Ballot()])
    b_list = p.ballots

    assert b_list == (Ballot(),)

    with pytest.raises(
        dataclasses.FrozenInstanceError, match="cannot assign to field 'ballots'"
    ):
        p.ballots = (Ballot(weight=5),)


def test_candidates_frozen():
    profile_no_cands = PreferenceProfile(
        ballots=[Ballot(ranking=[{"A"}, {"B"}]), Ballot(ranking=[{"C"}, {"B"}])]
    )
    assert set(profile_no_cands.candidates) == set(["A", "B", "C"])
    assert set(profile_no_cands.candidates_cast) == set(["A", "B", "C"])

    with pytest.raises(
        dataclasses.FrozenInstanceError, match="cannot assign to field 'candidates'"
    ):
        profile_no_cands.candidates = tuple()

    with pytest.raises(
        dataclasses.FrozenInstanceError,
        match="cannot assign to field 'candidates_cast'",
    ):
        profile_no_cands.candidates_cast = tuple()


def test_get_candidates_received_votes():
    profile_w_cands = PreferenceProfile(
        ballots=(
            Ballot(ranking=[{"A"}, {"B"}]),
            Ballot(ranking=[{"C"}, {"B"}]),
            Ballot(scores={"A": 4, "E": 4}),
        ),
        candidates=("A", "B", "C", "D", "E"),
    )
    vote_cands = profile_w_cands.candidates_cast
    all_cands = profile_w_cands.candidates

    assert set(all_cands) == {"A", "B", "C", "D", "E"}
    assert set(vote_cands) == {"A", "B", "C", "E"}


# def test_df_head():
#     profile = PreferenceProfile(
#         ballots=(
#             Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(1), scores={"A": 4}),
#             Ballot(ranking=({"B"}, {"C"}, {"E"}), weight=Fraction(2)),
#             Ballot(ranking=({"B", "C"}, {"E"}), weight=Fraction(1)),
#         )
#     )

#     true_df = pd.DataFrame(
#         {
#             "Ranking": [
#                 ("A", "B", "C"),
#                 ("B", "C", "E"),
#                 (f"{set({'B', 'C'})} (Tie)", "E"),
#             ],
#             "Scores": [("A:4.00",), tuple(), tuple()],
#             "Weight": [Fraction(1), Fraction(2), Fraction(1)],
#             "Percent": [f"{float(1/4):.2%}", f"{float(1/2):.2%}", f"{float(1/4):.2%}"],
#         }
#     )

#     true_df_totals = pd.DataFrame(
#         {
#             "Ranking": [("A", "B", "C"), ("B", "C", "E")],
#             "Scores": [("A:4.00",), tuple()],
#             "Weight": [Fraction(1), Fraction(2)],
#             "Percent": [f"{float(1/4):.2%}", f"{float(1/2):.2%}"],
#         },
#     )
#     true_df_totals.loc["Totals"] = {
#         "Ranking": "",
#         "Scores": "",
#         "Weight": f"{Fraction(3)} out of {Fraction(4)}",
#         "Percent": f"{float(75):.2f} out of 100%",
#     }
#     true_df_totals = true_df_totals.fillna("")

#     assert profile.head(2, percents=True).equals(
#         true_df.sort_values(by="Weight", ascending=False).reset_index(drop=True).head(2)
#     )
#     assert profile.head(2, percents=False).equals(
#         true_df[["Ranking", "Scores", "Weight"]]
#         .sort_values(by="Weight", ascending=False)
#         .reset_index(drop=True)
#         .head(2)
#     )
#     assert profile.head(2, sort_by_weight=False).equals(
#         true_df[["Ranking", "Scores", "Weight"]].head(2)
#     )

#     assert profile.head(2, sort_by_weight=False, totals=True, percents=True).equals(
#         true_df_totals
#     )


# def test_df_tail():
#     profile = PreferenceProfile(
#         ballots=(
#             Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(1)),
#             Ballot(ranking=({"B"}, {"C"}, {"E"}), weight=Fraction(2)),
#             Ballot(ranking=({"B", "C"}, {"E"}), weight=Fraction(1)),
#         )
#     )

#     true_df = pd.DataFrame(
#         {
#             "Ranking": [
#                 ("A", "B", "C"),
#                 ("B", "C", "E"),
#                 (f"{set({'B', 'C'})} (Tie)", "E"),
#             ],
#             "Scores": [tuple(), tuple(), tuple()],
#             "Weight": [Fraction(1), Fraction(2), Fraction(1)],
#             "Percent": [f"{float(1/4):.2%}", f"{float(1/2):.2%}", f"{float(1/4):.2%}"],
#         }
#     )

#     true_df_totals = pd.DataFrame(
#         {
#             "Ranking": [(f"{set({'B', 'C'})} (Tie)", "E"), ("B", "C", "E")],
#             "Scores": [tuple(), tuple()],
#             "Weight": [Fraction(1), Fraction(2)],
#             "Percent": [f"{float(1/4):.2%}", f"{float(2/4):.2%}"],
#         },
#     )
#     true_df_totals.index = [2, 1]
#     true_df_totals.loc["Totals"] = {
#         "Ranking": "",
#         "Scores": "",
#         "Weight": f"{Fraction(3)} out of {Fraction(4)}",
#         "Percent": f"{float(75):.2f} out of 100%",
#     }
#     true_df_totals = true_df_totals.fillna("")

#     true_df_sorted = true_df.sort_values(by="Weight", ascending=True)
#     true_df_sorted.index = [2, 1, 0]

#     assert profile.tail(2, percents=True).equals(true_df_sorted.head(2))
#     assert profile.tail(2, percents=False).equals(
#         true_df_sorted[["Ranking", "Scores", "Weight"]].head(2)
#     )
#     assert profile.tail(2, sort_by_weight=False).equals(
#         true_df[["Ranking", "Scores", "Weight"]].reindex(range(2, 0, -1)).tail(2)
#     )
#     assert profile.tail(2, sort_by_weight=False, totals=True, percents=True).equals(
#         true_df_totals
#     )
