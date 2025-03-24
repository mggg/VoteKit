from votekit.pref_profile import PreferenceProfile
from votekit.ballot import Ballot
from votekit.cleaning import remove_repeated_candidates
from fractions import Fraction
import pytest


def test_remove_repeated_candidates():
    ballot = Ballot(ranking=[{"A"}, {"A"}, {"B"}, {"C"}], weight=Fraction(1))
    ballot_tuple = (ballot, ballot)
    profile = PreferenceProfile(ballots=ballot_tuple)
    cleaned_profile = remove_repeated_candidates(profile)

    assert isinstance(cleaned_profile, PreferenceProfile)

    true_cleaned_ballot = Ballot(
        ranking=(frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    )

    assert cleaned_profile.ballots[0] == true_cleaned_ballot
    assert cleaned_profile != profile


def test_remove_repeated_candidates_ties():
    profile = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=[{"C", "A"}, {"A", "C"}, {"B"}],
            ),
            Ballot(
                ranking=[{"A", "C"}, {"C", "A"}, {"B"}],
            ),
        ]
    )
    clean_profile = remove_repeated_candidates(profile)
    true_clean = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=[{"C", "A"}, {"B"}],
            ),
            Ballot(
                ranking=[{"C", "A"}, {"B"}],
            ),
        ]
    )
    assert clean_profile.ballots == true_clean.ballots


def test_remove_repeated_cands_errors():
    with pytest.raises(TypeError, match="Ballot must have rankings:"):
        remove_repeated_candidates(PreferenceProfile(ballots=(Ballot(),)))

    with pytest.raises(TypeError, match="Ballot must only have rankings, not scores:"):
        remove_repeated_candidates(
            PreferenceProfile(
                ballots=(Ballot(ranking=(frozenset({"Chris"}),), scores={"Chris": 1}),)
            )
        )


def test_remove_repeated_cands_adjusted_count():
    profile = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=[{"C", "A"}, {"A", "C"}, {"B"}],
            ),
            Ballot(ranking=[{"A", "C"}, {"C", "A"}, {"B"}], weight=2),
            Ballot(
                ranking=[{"A", "C"}, {"B"}],
            ),
        ]
    )

    _, count = remove_repeated_candidates(profile, return_adjusted_count=True)

    assert count == 3
