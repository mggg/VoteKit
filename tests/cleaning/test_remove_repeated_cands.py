from votekit.pref_profile import PreferenceProfile
from votekit.ballot import Ballot
from votekit.cleaning import remove_repeated_candidates
from fractions import Fraction
import pytest


def test_remove_repeated_candidates():
    ballot = Ballot(ranking=[{"A"}, {"A"}, {"B"}, {"C"}], weight=Fraction(1))
    ballot_tuple = (ballot, ballot)
    profile = PreferenceProfile(ballots=ballot_tuple)
    cleaned_ballot = remove_repeated_candidates(ballot)
    cleaned_ballot_tuple = remove_repeated_candidates(ballot_tuple)
    cleaned_profile = remove_repeated_candidates(profile)

    assert isinstance(cleaned_ballot, Ballot)
    assert isinstance(cleaned_ballot_tuple, tuple)
    assert isinstance(cleaned_profile, PreferenceProfile)

    true_cleaned_ballot = Ballot(
        ranking=(frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    )
    assert cleaned_ballot == true_cleaned_ballot
    assert cleaned_ballot_tuple == (true_cleaned_ballot, true_cleaned_ballot)
    assert cleaned_profile.ballots == cleaned_ballot_tuple


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
        remove_repeated_candidates(Ballot())

    with pytest.raises(TypeError, match="Ballot must only have rankings, not scores:"):
        remove_repeated_candidates(
            Ballot(ranking=(frozenset({"Chris"}),), scores={"Chris": 1})
        )
