from votekit.pref_profile import PreferenceProfile, CleanedProfile
from votekit.ballot import Ballot
from votekit.cleaning import remove_cand
from fractions import Fraction

profile_no_ties = PreferenceProfile(
    ballots=[
        Ballot(ranking=[{"A"}, {"B"}], weight=1),
        Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=1 / 2),
        Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=3),
    ]
)

profile_with_ties = PreferenceProfile(
    ballots=[
        Ballot(ranking=[{"A", "B"}], weight=1),
        Ballot(ranking=[{"A", "B", "C"}], weight=1 / 2),
        Ballot(ranking=[{"A"}, {"C"}, {"B"}], weight=3),
    ]
)


def test_remove_cand():
    cleaned_profile = remove_cand("A", profile_no_ties)

    assert isinstance(cleaned_profile, CleanedProfile)
    assert cleaned_profile.parent_profile == profile_no_ties
    assert cleaned_profile.ballots == (
        Ballot(ranking=[{"B"}], weight=1),
        Ballot(ranking=[{"B"}, {"C"}], weight=1 / 2),
        Ballot(ranking=[{"C"}, {"B"}], weight=3),
    )
    assert cleaned_profile != profile_no_ties
    assert cleaned_profile.no_weight_alt_ballot_indices == []
    assert cleaned_profile.no_ranking_and_no_scores_alt_ballot_indices == []
    assert cleaned_profile.valid_but_alt_ballot_indices == [0, 1, 2]
    assert cleaned_profile.unalt_ballot_indices == []


def test_remove_mult_cands():
    cleaned_profile = remove_cand(["A", "B"], profile_no_ties)

    assert isinstance(cleaned_profile, CleanedProfile)
    assert cleaned_profile.parent_profile == profile_no_ties

    assert cleaned_profile.group_ballots().ballots == (
        Ballot(ranking=[{"C"}], weight=7 / 2),
    )
    assert cleaned_profile != profile_no_ties
    assert cleaned_profile.no_weight_alt_ballot_indices == []
    assert cleaned_profile.no_ranking_and_no_scores_alt_ballot_indices == [0]
    assert cleaned_profile.valid_but_alt_ballot_indices == [1, 2]
    assert cleaned_profile.unalt_ballot_indices == []


def test_remove_cand_with_ties():

    cleaned_profile = remove_cand(["A", "B"], profile_no_ties)
    assert isinstance(cleaned_profile, CleanedProfile)
    assert cleaned_profile.parent_profile == profile_no_ties

    assert cleaned_profile.group_ballots().ballots == (
        Ballot(ranking=[{"C"}], weight=7 / 2),
    )
    assert cleaned_profile != profile_with_ties
    assert cleaned_profile.no_weight_alt_ballot_indices == []
    assert cleaned_profile.no_ranking_and_no_scores_alt_ballot_indices == [0]
    assert cleaned_profile.valid_but_alt_ballot_indices == [1, 2]
    assert cleaned_profile.unalt_ballot_indices == []


def test_remove_cands_scores():
    profile = PreferenceProfile(
        ballots=(
            Ballot(
                ranking=({"A"}, {"B"}, {"C"}),
            ),
            Ballot(
                scores={"A": 3, "B": 2},
            ),
            Ballot(
                ranking=({"A"}, {"B"}, {"C"}),
                scores={"A": 3, "B": 2},
            ),
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(2)),
            Ballot(
                ranking=({"A"}, {"B"}, {"C"}),
                scores={"A": 3, "B": 2},
                weight=Fraction(2),
            ),
        ),
        candidates=("A", "B", "C"),
    )

    cleaned_profile = remove_cand("A", profile)
    assert isinstance(cleaned_profile, CleanedProfile)
    assert cleaned_profile.parent_profile == profile

    assert cleaned_profile.group_ballots().ballots == (
        Ballot(ranking=({"B"}, {"C"}), weight=Fraction(3)),
        Ballot(
            scores={"B": 2},
        ),
        Ballot(
            ranking=({"B"}, {"C"}),
            scores={"B": 2},
            weight=Fraction(3),
        ),
    )
    assert cleaned_profile != profile
    assert cleaned_profile.no_weight_alt_ballot_indices == []
    assert cleaned_profile.no_ranking_and_no_scores_alt_ballot_indices == []
    assert cleaned_profile.valid_but_alt_ballot_indices == [0, 1, 2, 3, 4]
    assert cleaned_profile.unalt_ballot_indices == []
