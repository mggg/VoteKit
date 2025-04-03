from fractions import Fraction
from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
import pytest

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
    assert rv[None] == Fraction(1)

    rv = profile.to_ranking_dict(standardize=True)
    assert rv[(frozenset({"A"}), frozenset({"B"}))] == Fraction(5, 11)
    assert rv[(frozenset({"C"}), frozenset({"B"}))] == Fraction(4, 11)
    assert rv[None] == Fraction(2,11)

def test_ranking_dict_warn():
    profile = PreferenceProfile(
        ballots=(
            Ballot(scores={"A": 4}),
        )
    )

    with pytest.warns(UserWarning, match=("You are trying to convert a profile that contains "
                           "no rankings to a ranking_dict.")):
        profile.to_ranking_dict()
    