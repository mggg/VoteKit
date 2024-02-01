from fractions import Fraction

from votekit.pref_profile import PreferenceProfile
from votekit.ballot import Ballot
from votekit.cleaning import remove_empty_ballots, deduplicate_profiles, remove_noncands


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
    assert ballot.ranking == ({"A"}, {"B"}, {"C"})


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
                ranking=[{"A"}, {"A"}, {"B"}, {"C"}],
                weight=Fraction(1),
                voter_set={"tom"},
            ),
            Ballot(
                ranking=[{"A"}, {"A"}, {"B"}, {"C"}],
                weight=Fraction(1),
                voter_set={"andy"},
            ),
        ]
    )
    profile_res = deduplicate_profiles(dirty)
    cleaned = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=[{"A"}, {"B"}, {"C"}],
                weight=Fraction(2),
                voter_set={"tom", "andy"},
            )
        ]
    )
    assert cleaned.ballots == profile_res.ballots


def test_deduplicate_mult_voters_diff_rankings():
    dirty = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=[{"A"}, {"A"}, {"B"}, {"B"}],
                weight=Fraction(1),
                voter_set={"tom"},
            ),
            Ballot(
                ranking=[{"A"}, {"A"}, {"B"}, {"C"}],
                weight=Fraction(1),
                voter_set={"andy"},
            ),
        ]
    )
    profile_res = deduplicate_profiles(dirty)
    cleaned = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"B"}], weight=Fraction(1), voter_set={"tom"}),
            Ballot(
                ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1), voter_set={"andy"}
            ),
        ]
    )
    assert cleaned.ballots == profile_res.ballots


def test_deduplicate_mult_voters_ties():
    dirty = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=[{"C", "A"}, {"A", "C"}, {"B"}],
                weight=Fraction(1),
                voter_set={"tom"},
            ),
            Ballot(
                ranking=[{"A", "C"}, {"C", "A"}, {"B"}],
                weight=Fraction(1),
                voter_set={"andy"},
            ),
        ]
    )
    profile_res = deduplicate_profiles(dirty)
    cleaned = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=[{"C", "A"}, {"B"}],
                weight=Fraction(2),
                voter_set={"tom", "andy"},
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


def test_remove_cands():
    dirty = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"Dog"}, {"B"}], weight=Fraction(1)),
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1)),
        ]
    )

    cleaned = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"B"}], weight=Fraction(1)),
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1)),
        ]
    )
    clean = remove_noncands(dirty, non_cands=["Dog"])

    assert clean.ballots == cleaned.ballots


def test_remove_none_ranking():
    dirty = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"Dog"}], weight=Fraction(1)),
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1)),
        ]
    )

    cleaned = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1)),
        ]
    )

    clean = remove_noncands(dirty, non_cands=["Dog"])
    assert clean.ballots == cleaned.ballots
