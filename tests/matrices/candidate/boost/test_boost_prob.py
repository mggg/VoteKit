from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
from votekit.matrices import boost_prob
import numpy as np

ballot_1 = Ballot(
    ranking=(frozenset({"Chris"}), frozenset({"Peter"}), frozenset({"Moon"}))
)
ballot_2 = Ballot(ranking=(frozenset({"Moon"}), frozenset({"Peter"})))
ballot_3 = Ballot(ranking=(frozenset({"Chris"}),))
pref_profile = PreferenceProfile(
    ballots=tuple(
        [ballot_1 for _ in range(5)]
        + [ballot_2 for _ in range(2)]
        + [ballot_3 for _ in range(1)]
    )
)


def test_boost_prob():
    cond, uncond = boost_prob("Chris", "Peter", pref_profile)

    assert cond == 5 / 7
    assert uncond == 3 / 4


def test_boost_prob_nan_no_mention():
    cond, uncond = boost_prob("Chris", "Mala", pref_profile)

    assert np.isnan(cond)
    assert uncond == 3 / 4


def test_boost_prob_nan_no_weight():
    cond, uncond = boost_prob("Chris", "Mala", PreferenceProfile())

    assert np.isnan(cond)
    assert np.isnan(uncond)
