from votekit.pref_profile import PreferenceProfile, CleanedProfile
from votekit.ballot import Ballot
from votekit.cleaning import remove_and_condense, remove_cand, condense_profile
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


def test_remove_and_condense():
    cleaned_profile = remove_and_condense("A", profile_no_ties)

    assert isinstance(cleaned_profile, CleanedProfile)
    assert cleaned_profile.parent_profile == profile_no_ties
    assert cleaned_profile.ballots == (
        Ballot(ranking=[{"B"}], weight=1),
        Ballot(ranking=[{"B"}, {"C"}], weight=1 / 2),
        Ballot(ranking=[{"C"}, {"B"}], weight=3),
    )
    assert cleaned_profile != profile_no_ties
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_no_score_altr_idxs == set()
    assert cleaned_profile.nonempty_altr_idxs == {0, 1, 2}
    assert cleaned_profile.unaltr_idxs == set()


def test_remove_then_condense_equivalence():
    cleaned_profile_1 = remove_and_condense("A", profile_no_ties)
    cleaned_profile_2 = condense_profile(remove_cand("A", profile_no_ties))

    assert cleaned_profile_1 == cleaned_profile_2


def test_remove_mult_cands():
    cleaned_profile = remove_and_condense(["A", "B"], profile_no_ties)

    assert isinstance(cleaned_profile, CleanedProfile)
    assert cleaned_profile.parent_profile == profile_no_ties

    assert set(cleaned_profile.group_ballots().ballots) == set(
        [
            # Ballot(ranking=[frozenset(), frozenset()], weight=1),
            Ballot(ranking=[{"C"}], weight=7 / 2),
        ]
    )
    assert cleaned_profile != profile_no_ties
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_no_score_altr_idxs == {0}
    assert cleaned_profile.nonempty_altr_idxs == {1, 2}
    assert cleaned_profile.unaltr_idxs == set()


def test_remove_and_condense_with_ties():

    cleaned_profile = remove_and_condense(["A", "B"], profile_with_ties)
    assert isinstance(cleaned_profile, CleanedProfile)
    assert cleaned_profile.parent_profile == profile_with_ties

    assert set(cleaned_profile.group_ballots().ballots) == set(
        [
            # Ballot(ranking=[frozenset()], weight=1),
            Ballot(ranking=[{"C"}], weight=7 / 2),
        ]
    )
    assert cleaned_profile != profile_with_ties
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_no_score_altr_idxs == {0}
    assert cleaned_profile.nonempty_altr_idxs == {1, 2}
    assert cleaned_profile.unaltr_idxs == set()


def test_remove_and_condense_scores():
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

    cleaned_profile = remove_and_condense("A", profile)
    assert isinstance(cleaned_profile, CleanedProfile)
    assert cleaned_profile.parent_profile == profile

    assert cleaned_profile.ballots == (
        Ballot(
            ranking=({"B"}, {"C"}),
        ),
        Ballot(
            scores={"B": 2},
        ),
        Ballot(
            ranking=({"B"}, {"C"}),
            scores={"B": 2},
        ),
        Ballot(ranking=({"B"}, {"C"}), weight=Fraction(2)),
        Ballot(
            ranking=({"B"}, {"C"}),
            scores={"B": 2},
            weight=Fraction(2),
        ),
    )
    assert cleaned_profile != profile
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_no_score_altr_idxs == set()
    assert cleaned_profile.nonempty_altr_idxs == {0, 1, 2, 3, 4}
    assert cleaned_profile.unaltr_idxs == set()
