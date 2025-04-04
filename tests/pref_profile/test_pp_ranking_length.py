from fractions import Fraction
import pandas as pd
from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
import pytest


def test_ranking_length_default():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C", "D"})),
            Ballot(ranking=({"A"}, {"B"}), weight=Fraction(3, 2)),
            Ballot(ranking=({"C"}, {"B"}), scores={"A": 4}, weight=2),
            Ballot(scores={"A": 4}),
        )
    )

    assert profile.max_ranking_length == 3


def test_ranking_length_warning():

    with pytest.warns(
        UserWarning,
        match=(
            "Profile does not contain rankings but "
            f"max_ranking_length=3. Setting max_ranking_length"
            " to 0."
        ),
    ):
        profile_scores = PreferenceProfile(
            ballots=(
                Ballot(
                    scores={"A": 2, "B": 4, "D": 1},
                ),
            ),
            candidates=["A", "B", "C", "D", "E"],
            max_ranking_length=3,
        )

    assert profile_scores.max_ranking_length == 0


def test_ranking_length_no_default():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C", "D"})),
            Ballot(ranking=({"A"}, {"B"}), weight=Fraction(3, 2)),
            Ballot(ranking=({"C"}, {"B"}), scores={"A": 4}, weight=2),
            Ballot(scores={"A": 4}),
        ),
        max_ranking_length=4,
    )

    assert profile.max_ranking_length == 4
