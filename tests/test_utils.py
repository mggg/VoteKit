from fractions import Fraction

from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
from votekit.utils import (
    mentions,
    first_place_votes,
    borda_scores,
    scores_into_set_list,
)


profile = PreferenceProfile(
    ballots=[
        Ballot(ranking=[{"A"}, {"B"}], weight=Fraction(1), voter_set={"tom"}),
        Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1), voter_set={"andy"}),
        Ballot(ranking=[{"A"}, {"C"}, {"B"}], weight=Fraction(3), voter_set={"andy"}),
    ]
)


def test_first_place_votes():
    correct = {"A": 5, "B": 0, "C": 0}
    test = first_place_votes(profile)
    assert correct == test


def test_first_place_ties():
    ties = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A", "B"}, {"D"}], weight=Fraction(1), voter_set={"tom"}),
            Ballot(
                ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1), voter_set={"andy"}
            ),
            Ballot(
                ranking=[{"A"}, {"C"}, {"B"}], weight=Fraction(3), voter_set={"andy"}
            ),
        ]
    )
    correct = {"A": 4.5, "B": 0.5, "C": 0, "D": 0}
    test = first_place_votes(ties)
    assert test == correct


def test_mentions():
    correct = {"A": 5, "B": 5, "C": 4}
    test = mentions(profile)
    assert correct == test


def test_mentions_with_ties():
    ties = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"B", "D"}], weight=Fraction(1), voter_set={"tom"}),
            Ballot(
                ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1), voter_set={"andy"}
            ),
            Ballot(
                ranking=[{"A"}, {"C"}, {"B"}], weight=Fraction(3), voter_set={"andy"}
            ),
        ]
    )
    correct = {"A": 5, "B": 4.5, "C": 4, "D": 0.5}
    test = mentions(ties)
    assert test == correct


ballot_list = [
    Ballot(
        id=None, ranking=[{"A"}, {"C"}, {"D"}, {"B"}, {"E"}], weight=Fraction(10, 1)
    ),
    Ballot(
        id=None, ranking=[{"A"}, {"B"}, {"C"}, {"D"}, {"E"}], weight=Fraction(10, 1)
    ),
    Ballot(
        id=None, ranking=[{"D"}, {"A"}, {"E"}, {"B"}, {"C"}], weight=Fraction(10, 1)
    ),
    Ballot(id=None, ranking=[{"A"}], weight=Fraction(24, 1)),
]
TEST_PROFILE = PreferenceProfile(ballots=ballot_list)


def test_borda_long_ballot():
    target_borda_dict = {
        "A": Fraction(260),
        "B": Fraction(140),
        "C": Fraction(140),
        "D": Fraction(160),
        "E": Fraction(110),
    }
    method_borda_dict = borda_scores(TEST_PROFILE)
    assert method_borda_dict == target_borda_dict


def test_borda_short_ballot():
    method_borda_dict = borda_scores(TEST_PROFILE, ballot_length=3)
    target_borda_dict = {
        "A": Fraction(152),
        "B": Fraction(38),
        "C": Fraction(48),
        "D": Fraction(58),
        "E": Fraction(28),
    }

    assert method_borda_dict == target_borda_dict


# def test_candidate_position_dict():
#     assert True


# def test_tiebreak_none():
#     assert True


# def test_tiebreak_random():
#     assert True


# def test_tiebreak_firstplace():
#     assert True


# def test_tiebreak_borda():
#     assert True


def test_scores_into_set_list():
    score_dict = {"A": 3, "B": 2, "C": 3, "D": 2, "E": -1, "F": 2.5}
    set_order = scores_into_set_list(score_dict)
    target_order = [{"A", "C"}, {"F"}, {"B", "D"}, {"E"}]
    assert set_order == target_order


# def test_elect_cands_from_set_ranking():
#     assert True
