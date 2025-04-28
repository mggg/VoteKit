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
        Ballot(ranking=[frozenset(), {"B"}], weight=1),
        Ballot(ranking=[frozenset(), {"B"}, {"C"}], weight=1 / 2),
        Ballot(ranking=[{"C"}, {"B"}, frozenset()], weight=3),
    )
    assert cleaned_profile != profile_no_ties
    assert cleaned_profile.no_weight_altr_ballot_indices == set()
    assert cleaned_profile.no_ranking_and_no_scores_altr_ballot_indices == set()
    assert cleaned_profile.valid_but_altr_ballot_indices == {0, 1, 2}
    assert cleaned_profile.unaltr_ballot_indices == set()


def test_remove_mult_cands():
    cleaned_profile = remove_cand(["A", "B"], profile_no_ties)

    assert isinstance(cleaned_profile, CleanedProfile)
    assert cleaned_profile.parent_profile == profile_no_ties

    assert set(cleaned_profile.group_ballots().ballots) == set(
        [
            Ballot(ranking=[frozenset(), frozenset()], weight=1),
            Ballot(ranking=[frozenset(), frozenset(), {"C"}], weight=1 / 2),
            Ballot(ranking=[{"C"}, frozenset(), frozenset()], weight=3),
        ]
    )
    assert cleaned_profile != profile_no_ties
    assert cleaned_profile.no_weight_altr_ballot_indices == set()
    assert cleaned_profile.no_ranking_and_no_scores_altr_ballot_indices == set()
    assert cleaned_profile.valid_but_altr_ballot_indices == {0, 1, 2}
    assert cleaned_profile.unaltr_ballot_indices == set()


def test_remove_cand_with_ties():

    cleaned_profile = remove_cand(["A", "B"], profile_with_ties)
    assert isinstance(cleaned_profile, CleanedProfile)
    assert cleaned_profile.parent_profile == profile_with_ties

    assert set(cleaned_profile.group_ballots().ballots) == set(
        [
            Ballot(ranking=[frozenset()], weight=1),
            Ballot(ranking=[{"C"}], weight=1 / 2),
            Ballot(ranking=[frozenset(), {"C"}, frozenset()], weight=3),
        ]
    )
    assert cleaned_profile != profile_with_ties
    assert cleaned_profile.no_weight_altr_ballot_indices == set()
    assert cleaned_profile.no_ranking_and_no_scores_altr_ballot_indices == set()
    assert cleaned_profile.valid_but_altr_ballot_indices == {0, 1, 2}
    assert cleaned_profile.unaltr_ballot_indices == set()


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

    assert cleaned_profile.ballots == (
        Ballot(
            ranking=(frozenset(), {"B"}, {"C"}),
        ),
        Ballot(
            scores={"B": 2},
        ),
        Ballot(
            ranking=(frozenset(), {"B"}, {"C"}),
            scores={"B": 2},
        ),
        Ballot(ranking=(frozenset(), {"B"}, {"C"}), weight=Fraction(2)),
        Ballot(
            ranking=(frozenset(), {"B"}, {"C"}),
            scores={"B": 2},
            weight=Fraction(2),
        ),
    )
    assert cleaned_profile != profile
    assert cleaned_profile.no_weight_altr_ballot_indices == set()
    assert cleaned_profile.no_ranking_and_no_scores_altr_ballot_indices == set()
    assert cleaned_profile.valid_but_altr_ballot_indices == {0, 1, 2, 3, 4}
    assert cleaned_profile.unaltr_ballot_indices == set()
