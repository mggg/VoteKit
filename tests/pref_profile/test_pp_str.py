from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
import pandas as pd
import numpy as np
from fractions import Fraction

ballots_scores = [
    Ballot(
        weight=2,
        scores={
            "A": 1,
            "B": 2,
        },
    ),
    Ballot(scores={"D": 2, "E": 1}, id="X29", voter_set={"Chris"}),
    Ballot(),
    Ballot(weight=0),
]

ballots_rankings = [
    Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
    Ballot(ranking=({"A", "B"}, frozenset(), {"D"}), id="X29", voter_set={"Chris"}),
    Ballot(),
    Ballot(weight=0),
]

ballots_mixed = [
    Ballot(
        weight=2,
        scores={
            "A": 1,
            "B": 2,
        },
    ),
    Ballot(ranking=({"A", "B"}, frozenset(), {"D"}), id="X29", voter_set={"Chris"}),
    Ballot(
        ranking=({"A"}, {"B"}, {"C"}),
        weight=2,
        scores={
            "A": 1,
            "B": 2,
        },
    ),
]

def test_print_profile_rankings():
    print(PreferenceProfile(ballots = ballots_rankings))

def test_print_profile_scores():
    print(PreferenceProfile(ballots = ballots_scores))

def test_print_profile_mixed():
    print(PreferenceProfile(ballots = ballots_mixed))