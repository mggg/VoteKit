from votekit.profile import PreferenceProfile
from votekit.ballot import Ballot
from votekit.cleaning import remove_empty_ballots, deduplicate_profiles
from fractions import Fraction


def test_remove_empty_ballots():
    profile = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1)),
            Ballot(ranking=[], weight=Fraction(1)),
        ]
    )
    profile_cleaned = remove_empty_ballots(profile)
    assert len(profile_cleaned.get_ballots()) == 1
    ballot = profile_cleaned.get_ballots()[0]
    assert ballot.ranking == [{"A"}, {"B"}, {"C"}]


def test_deduplicate_single():
    dirty = PreferenceProfile(
        ballots=[Ballot(ranking=[{"A"}, {"A"}, {"B"}, {"C"}], weight=Fraction(1))]
    )
    profile_res = deduplicate_profiles(dirty)
    cleaned = PreferenceProfile(
        ballots=[Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1))]
    )
    assert cleaned.ballots == profile_res.ballots


def test_deduplicate_mult_voters_same_rankings():
    dirty = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=[{"A"}, {"A"}, {"B"}, {"C"}], weight=Fraction(1), voters={"tom"}
            ),
            Ballot(
                ranking=[{"A"}, {"A"}, {"B"}, {"C"}],
                weight=Fraction(1),
                voters={"andy"},
            ),
        ]
    )
    profile_res = deduplicate_profiles(dirty)
    cleaned = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=[{"A"}, {"B"}, {"C"}],
                weight=Fraction(2),
                voters={"tom", "andy"},
            )
        ]
    )
    assert cleaned.ballots == profile_res.ballots


def test_deduplicate_mult_voters_diff_rankings():
    dirty = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=[{"A"}, {"A"}, {"B"}, {"B"}], weight=Fraction(1), voters={"tom"}
            ),
            Ballot(
                ranking=[{"A"}, {"A"}, {"B"}, {"C"}],
                weight=Fraction(1),
                voters={"andy"},
            ),
        ]
    )
    profile_res = deduplicate_profiles(dirty)
    cleaned = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"B"}], weight=Fraction(1), voters={"tom"}),
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1), voters={"andy"}),
        ]
    )
    assert cleaned.ballots == profile_res.ballots


def test_deduplicate_mult_voters_ties():
    dirty = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=[{"C", "A"}, {"A", "C"}, {"B"}],
                weight=Fraction(1),
                voters={"tom"},
            ),
            Ballot(
                ranking=[{"A", "C"}, {"C", "A"}, {"B"}],
                weight=Fraction(1),
                voters={"andy"},
            ),
        ]
    )
    profile_res = deduplicate_profiles(dirty)
    cleaned = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=[{"C", "A"}, {"B"}], weight=Fraction(2), voters={"tom", "andy"}
            )
        ]
    )
    assert cleaned.ballots == profile_res.ballots


def test_deduplicate_no_voters_diff_rankings():
    dirty = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"A"}, {"B"}, {"B"}], weight=Fraction(1)),
            Ballot(ranking=[{"A"}, {"A"}, {"B"}, {"C"}], weight=Fraction(1)),
        ]
    )
    profile_res = deduplicate_profiles(dirty)
    cleaned = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"B"}], weight=Fraction(1)),
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1)),
        ]
    )
    assert cleaned.ballots == profile_res.ballots
