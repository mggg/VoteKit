from fractions import Fraction
from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
import pytest

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
    assert rv[None] == Fraction(5, 2)

    rv = profile.to_scores_dict(standardize=True)
    assert rv[(("A", Fraction(4)),)] == Fraction(6,11)
    assert rv[None] == Fraction(5, 11)

def test_scores_dict_warn():
    profile = PreferenceProfile(
        ballots=(
                        Ballot(ranking=({"A"}, {"B"})),

        )
    )

    with pytest.warns(UserWarning, match=("You are trying to convert a profile that contains "
                           "no scores to a scores_dict.")):
        profile.to_scores_dict()
    