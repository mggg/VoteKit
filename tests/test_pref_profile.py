from fractions import Fraction
import pandas as pd
from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
import pytest
import dataclasses


def test_init():
    empty_profile = PreferenceProfile()
    assert empty_profile.ballots == ()
    assert not empty_profile.candidates
    print(empty_profile.df)
    assert empty_profile.df.equals(
        pd.DataFrame({"Ranking": [], "Scores": [], "Weight": [], "Percent": []})
    )
    assert not empty_profile.candidates_cast
    assert not empty_profile.total_ballot_wt
    assert not empty_profile.num_ballots


def test_unique_cands_validator():
    with pytest.raises(ValueError, match="All candidates must be unique."):
        PreferenceProfile(candidates=("A", "A", "B"))

    PreferenceProfile(candidates=("A", "B"))


def test_ballots():
    p = PreferenceProfile(ballots=[Ballot()])
    b_list = p.ballots

    assert b_list == (Ballot(),)

    with pytest.raises(
        dataclasses.FrozenInstanceError, match="cannot assign to field 'ballots'"
    ):
        p.ballots = (Ballot(weight=5),)


def test_candidates():
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


def test_get_num_ballots():
    # should count duplicates, not weight
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"})),
            Ballot(ranking=({"A"}, {"B"})),
            Ballot(ranking=({"C"}, {"B"}), weight=2),
        )
    )
    assert profile.num_ballots == 3

    with pytest.raises(
        dataclasses.FrozenInstanceError, match="cannot assign to field 'num_ballots'"
    ):
        profile.num_ballots = 0


def test_total_ballot_wt():
    # should count weight
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"})),
            Ballot(ranking=({"A"}, {"B"}), weight=Fraction(3, 2)),
            Ballot(ranking=({"C"}, {"B"}), weight=2),
        )
    )
    assert profile.total_ballot_wt == Fraction(9, 2)
    with pytest.raises(
        dataclasses.FrozenInstanceError,
        match="cannot assign to field 'total_ballot_wt'",
    ):
        profile.total_ballot_wt = 0


def test_to_ballot_dict():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"})),
            Ballot(ranking=({"A"}, {"B"}), weight=Fraction(3, 2)),
            Ballot(ranking=({"C"}, {"B"}), scores={"A": 4}, weight=2),
            Ballot(scores={"A": 4}),
        )
    )
    rv = profile.to_ballot_dict(standardize=False)
    assert rv[Ballot(ranking=({"A"}, {"B"}))] == Fraction(5, 2)
    assert rv[Ballot(ranking=({"C"}, {"B"}), scores={"A": 4})] == Fraction(2, 1)
    assert rv[Ballot(scores={"A": 4})] == Fraction(1)


def test_to_ranking_dict():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"})),
            Ballot(ranking=({"A"}, {"B"}), weight=Fraction(3, 2)),
            Ballot(ranking=({"C"}, {"B"}), scores={"A": 4}, weight=2),
            Ballot(scores={"A": 4}),
        )
    )
    rv = profile.to_ranking_dict(standardize=False)
    assert rv[(frozenset({"A"}), frozenset({"B"}))] == Fraction(5, 2)
    assert rv[(frozenset({"C"}), frozenset({"B"}))] == Fraction(2, 1)
    assert rv[(frozenset(),)] == Fraction(1)


def test_to_scores_dict():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"})),
            Ballot(ranking=({"A"}, {"B"}), weight=Fraction(3, 2)),
            Ballot(ranking=({"C"}, {"B"}), scores={"A": 4}, weight=2),
            Ballot(scores={"A": 4}),
        )
    )
    rv = profile.to_scores_dict(standardize=False)
    assert rv[(("A", Fraction(4)),)] == Fraction(3)
    assert rv[tuple()] == Fraction(5, 2)


def test_condense_profile_ranking():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(1)),
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(2)),
        ),
        candidates=("A", "B", "C", "D"),
    )
    pp = profile.condense_ballots()
    assert pp.ballots[0] == Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(3))
    assert set(pp.candidates) == set(profile.candidates)


def test_condense_profile_scores():
    profile = PreferenceProfile(
        ballots=(
            Ballot(
                ranking=({"A"}, {"B"}, {"C"}),
                scores={"A": 3, "B": 2},
                weight=Fraction(1),
            ),
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(2)),
            Ballot(
                ranking=({"A"}, {"B"}, {"C"}),
                scores={"A": 3, "B": 2},
                weight=Fraction(2),
            ),
        )
    )
    pp = profile.condense_ballots()
    assert pp.ballots[0] == Ballot(
        ranking=({"A"}, {"B"}, {"C"}), scores={"A": 3, "B": 2}, weight=Fraction(3)
    )


def test_profile_equals():
    profile1 = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(1)),
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(2)),
            Ballot(ranking=({"E"}, {"C"}, {"B"}), weight=Fraction(2)),
        )
    )
    profile2 = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"E"}, {"C"}, {"B"}), weight=Fraction(2)),
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(3)),
        )
    )
    assert profile1 == profile2


def test_create_df():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(2)),
            Ballot(ranking=({"B"}, {"C"}, {"E"}), weight=Fraction(1)),
            Ballot(ranking=({"B", "C"}, {"E"}), weight=Fraction(1)),
        )
    )

    true_df = pd.DataFrame(
        {
            "Ranking": [
                ("A", "B", "C"),
                ("B", "C", "E"),
                (f"{set({'B', 'C'})} (Tie)", "E"),
            ],
            "Scores": [tuple(), tuple(), tuple()],
            "Weight": [Fraction(2), Fraction(1), Fraction(1)],
            "Percent": [f"{float(1/2):.2%}", f"{float(1/4):.2%}", f"{float(1/4):.2%}"],
        }
    )

    assert profile.df.equals(true_df)


def test_vote_share_with_zeros():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(0)),
            Ballot(ranking=({"B"}, {"C"}, {"E"}), weight=Fraction(0)),
        )
    )
    assert sum(profile.df["Weight"]) == 0


def test_df_head():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(1), scores={"A": 4}),
            Ballot(ranking=({"B"}, {"C"}, {"E"}), weight=Fraction(2)),
            Ballot(ranking=({"B", "C"}, {"E"}), weight=Fraction(1)),
        )
    )

    true_df = pd.DataFrame(
        {
            "Ranking": [
                ("A", "B", "C"),
                ("B", "C", "E"),
                (f"{set({'B', 'C'})} (Tie)", "E"),
            ],
            "Scores": [("A:4.00",), tuple(), tuple()],
            "Weight": [Fraction(1), Fraction(2), Fraction(1)],
            "Percent": [f"{float(1/4):.2%}", f"{float(1/2):.2%}", f"{float(1/4):.2%}"],
        }
    )

    true_df_totals = pd.DataFrame(
        {
            "Ranking": [("A", "B", "C"), ("B", "C", "E")],
            "Scores": [("A:4.00",), tuple()],
            "Weight": [Fraction(1), Fraction(2)],
            "Percent": [f"{float(1/4):.2%}", f"{float(1/2):.2%}"],
        },
    )
    true_df_totals.loc["Totals"] = {
        "Ranking": "",
        "Scores": "",
        "Weight": f"{Fraction(3)} out of {Fraction(4)}",
        "Percent": f"{float(75):.2f} out of 100%",
    }
    true_df_totals = true_df_totals.fillna("")

    assert profile.head(2, percents=True).equals(
        true_df.sort_values(by="Weight", ascending=False).reset_index(drop=True).head(2)
    )
    assert profile.head(2, percents=False).equals(
        true_df[["Ranking", "Scores", "Weight"]]
        .sort_values(by="Weight", ascending=False)
        .reset_index(drop=True)
        .head(2)
    )
    assert profile.head(2, sort_by_weight=False).equals(
        true_df[["Ranking", "Scores", "Weight"]].head(2)
    )

    assert profile.head(2, sort_by_weight=False, totals=True, percents=True).equals(
        true_df_totals
    )


def test_df_tail():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(1)),
            Ballot(ranking=({"B"}, {"C"}, {"E"}), weight=Fraction(2)),
            Ballot(ranking=({"B", "C"}, {"E"}), weight=Fraction(1)),
        )
    )

    true_df = pd.DataFrame(
        {
            "Ranking": [
                ("A", "B", "C"),
                ("B", "C", "E"),
                (f"{set({'B', 'C'})} (Tie)", "E"),
            ],
            "Scores": [tuple(), tuple(), tuple()],
            "Weight": [Fraction(1), Fraction(2), Fraction(1)],
            "Percent": [f"{float(1/4):.2%}", f"{float(1/2):.2%}", f"{float(1/4):.2%}"],
        }
    )

    true_df_totals = pd.DataFrame(
        {
            "Ranking": [(f"{set({'B', 'C'})} (Tie)", "E"), ("B", "C", "E")],
            "Scores": [tuple(), tuple()],
            "Weight": [Fraction(1), Fraction(2)],
            "Percent": [f"{float(1/4):.2%}", f"{float(2/4):.2%}"],
        },
    )
    true_df_totals.index = [2, 1]
    true_df_totals.loc["Totals"] = {
        "Ranking": "",
        "Scores": "",
        "Weight": f"{Fraction(3)} out of {Fraction(4)}",
        "Percent": f"{float(75):.2f} out of 100%",
    }
    true_df_totals = true_df_totals.fillna("")

    true_df_sorted = true_df.sort_values(by="Weight", ascending=True)
    true_df_sorted.index = [2, 1, 0]

    assert profile.tail(2, percents=True).equals(true_df_sorted.head(2))
    assert profile.tail(2, percents=False).equals(
        true_df_sorted[["Ranking", "Scores", "Weight"]].head(2)
    )
    assert profile.tail(2, sort_by_weight=False).equals(
        true_df[["Ranking", "Scores", "Weight"]].reindex(range(2, 0, -1)).tail(2)
    )
    assert profile.tail(2, sort_by_weight=False, totals=True, percents=True).equals(
        true_df_totals
    )


def test_add_profiles():
    profile_1 = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A", "B", "C"},), weight=1),
            Ballot(ranking=({"A", "B", "C"},), weight=2),
            Ballot(ranking=({"B", "A", "C"},), weight=1),
        )
    )

    profile_2 = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A", "B", "C"},), weight=1),
            Ballot(ranking=({"C", "B", "A"},), weight=47),
        )
    )

    summed_profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A", "B", "C"},), weight=4),
            Ballot(ranking=({"B", "A", "C"},), weight=1),
            Ballot(ranking=({"C", "B", "A"},), weight=47),
        )
    )

    assert profile_1 + profile_2 == summed_profile


def test_str():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"})),
            Ballot(ranking=({"A"}, {"B"}), weight=Fraction(3, 2)),
            Ballot(ranking=({"C"}, {"B"}), weight=2),
        )
    )
    print(profile)


def test_csv():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"})),
            Ballot(ranking=({"A"}, {"B"}), weight=Fraction(3, 2)),
            Ballot(ranking=({"C", "B"},), weight=2),
        )
    )
    fpath = "tests/data/csv/test_pref_profile_to_csv.csv"
    profile.to_csv(fpath)
