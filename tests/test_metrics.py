from votekit.metrics import borda_scores
from votekit.ballot import Ballot
from votekit.profile import PreferenceProfile

from fractions import Fraction

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
