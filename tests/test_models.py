from fractions import Fraction

from votekit.models import fix_ties, recursively_fix_ties
from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
from votekit.elections.election_types import STV
from votekit.elections.transfers import fractional_transfer


def test_single_tie():
    tied = Ballot(ranking=[{"A"}, {"B", "D"}, {"C"}], weight=Fraction(4))
    resolved = [
        Ballot(ranking=[{"A"}, {"B"}, {"D"}, {"C"}], weight=Fraction(2)),
        Ballot(ranking=[{"A"}, {"D"}, {"B"}, {"C"}], weight=Fraction(2)),
    ]

    test = fix_ties(tied)
    # order of permuted ballots is stochastic so we can't test if the two lists
    # are equal
    assert set(resolved) == (set(test))
    assert len(test) == 2


def test_tie_for_last():
    tied = Ballot(ranking=[{"A"}, {"B"}, {"D"}, {"C", "E"}], weight=Fraction(2, 1))
    resolved = [
        Ballot(ranking=[{"A"}, {"B"}, {"D"}, {"C"}, {"E"}], weight=Fraction(1)),
        Ballot(ranking=[{"A"}, {"B"}, {"D"}, {"E"}, {"C"}], weight=Fraction(1)),
    ]
    test = fix_ties(tied)

    assert set(test) == set(resolved)


def test_multiple_ties():
    tied = Ballot(ranking=[{"A"}, {"B", "D"}, {"C", "E"}], weight=Fraction(4))
    part = fix_ties(tied)
    complete = recursively_fix_ties(part, 2)

    assert len(complete) == 4
    assert (
        Ballot(ranking=[{"A"}, {"B"}, {"D"}, {"C"}, {"E"}], weight=Fraction(1))
        in complete
    )


def test_all_ties():
    tied = Ballot(ranking=[{"A", "F"}, {"B", "D"}, {"C", "E"}], weight=Fraction(4))
    part = fix_ties(tied)
    complete = recursively_fix_ties(part, 3)

    assert len(complete) == 8
    assert (
        Ballot(
            ranking=[{"A"}, {"F"}, {"B"}, {"D"}, {"C"}, {"E"}], weight=Fraction(1, 2)
        )
        in complete
    )


def test_resolve_ties_STV():
    profile = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=Fraction(3)),
            Ballot(ranking=[{"C", "A"}, {"B", "D"}], weight=Fraction(2)),
        ]
    )

    election = STV(profile, seats=1, transfer=fractional_transfer, ballot_ties=True)
    out = election.state.profile
    correct = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=Fraction(3)),
            Ballot(ranking=[{"C"}, {"A"}, {"B"}, {"D"}], weight=Fraction(1, 2)),
            Ballot(ranking=[{"C"}, {"A"}, {"D"}, {"B"}], weight=Fraction(1, 2)),
            Ballot(ranking=[{"A"}, {"C"}, {"D"}, {"B"}], weight=Fraction(1, 2)),
            Ballot(ranking=[{"A"}, {"C"}, {"B"}, {"D"}], weight=Fraction(1, 2)),
        ]
    )

    assert correct == out
    assert len(out.get_ballots()) == 5


def test_allow_ties_STV():
    profile = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=Fraction(3)),
            Ballot(ranking=[{"C", "A"}, {"B", "D"}], weight=Fraction(2)),
        ]
    )

    election = STV(profile, seats=1, transfer=fractional_transfer, ballot_ties=False)
    out = election.state.profile
    assert profile == out
