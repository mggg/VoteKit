from fractions import Fraction
from pathlib import Path

from votekit.ballot import Ballot
from votekit.cvr_loaders import load_csv
from votekit.utils import remove_cand
from votekit.pref_profile import PreferenceProfile


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/csv/"

test_profile = load_csv(DATA_DIR / "test_election_A.csv")
mn_profile = load_csv("src/votekit/data/mn_2013_cast_vote_record.csv")


def test_unique_cands():
    profile = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1)),
            Ballot(ranking=[{"B"}, {"C"}, {"E"}], weight=Fraction(1)),
        ]
    )
    cands = profile.get_candidates()
    unique_cands = {"A", "B", "C", "E"}
    assert unique_cands == set(cands)
    assert len(cands) == len(unique_cands)


def test_updates_not_in_place():
    before = test_profile.get_ballots()
    remove = "a"
    remove_cand(remove, before)
    assert remove in test_profile.get_candidates()


def test_to_dict():
    rv = test_profile.to_dict(standardize=False)
    assert rv[("a", "b", "c")] == Fraction(2, 1)
    assert rv[("b", "a", "e")] == Fraction(1, 1)


def test_condense_profile():
    profile = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1)),
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(2)),
        ]
    )
    profile.condense_ballots()
    assert profile.ballots[0] == Ballot(
        ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(3)
    )


def test_profile_equals():
    profile1 = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1)),
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(2)),
            Ballot(ranking=[{"E"}, {"C"}, {"B"}], weight=Fraction(2)),
        ]
    )
    profile2 = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"E"}, {"C"}, {"B"}], weight=Fraction(2)),
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(3)),
        ]
    )
    assert profile1 == profile2


def test_create_df():
    profile = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1)),
            Ballot(ranking=[{"B"}, {"C"}, {"E"}], weight=Fraction(1)),
        ]
    )
    df = profile._create_df()
    assert len(df) == 2


def test_vote_share_with_zeros():
    profile = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(0)),
            Ballot(ranking=[{"B"}, {"C"}, {"E"}], weight=Fraction(0)),
        ]
    )
    df = profile._create_df()
    assert sum(df["Voter Share"]) == 0


def test_df_percents():
    profile = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1)),
            Ballot(ranking=[{"B"}, {"C"}, {"E"}], weight=Fraction(1)),
        ]
    )
    rv = profile.head(2, percents=True)
    assert "Voter Share" in rv
    rv = profile.head(2)
    assert "Voter Share" not in rv
