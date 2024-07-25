from fractions import Fraction
from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
from votekit.utils import (
    ballots_by_first_cand,
    remove_cand,
    add_missing_cands,
    validate_score_vector,
    score_profile,
    first_place_votes,
    mentions,
    borda_scores,
    tie_broken_ranking,
    score_dict_to_ranking,
    elect_cands_from_set_ranking,
    expand_tied_ballot,
    resolve_profile_ties,
)
import pytest

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

profile_with_missing = PreferenceProfile(
    ballots=[
        Ballot(ranking=[{"A", "B"}, {"D"}], weight=1),
        Ballot(ranking=[{"A", "B", "C", "D"}], weight=1 / 2),
        Ballot(ranking=[{"A"}, {"C"}, {"B"}], weight=3),
        Ballot(ranking=[{"A"}, {"C"}, {"B"}, {"D"}, {"E"}]),
    ],
    candidates=["A", "B", "C", "D", "E"],
)


def test_ballots_by_first_cand():
    cand_dict = ballots_by_first_cand(profile_no_ties)
    partition = {
        "A": [
            Ballot(ranking=[{"A"}, {"B"}], weight=1),
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=1 / 2),
        ],
        "B": [],
        "C": [Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=3)],
    }

    assert cand_dict == partition


def test_ballots_by_first_cand_error():
    with pytest.raises(ValueError, match="has a tie for first."):
        ballots_by_first_cand(profile_with_ties)


def test_remove_cand_dif_types():
    no_a_true = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"B"}], weight=1),
            Ballot(ranking=[{"B"}, {"C"}], weight=1 / 2),
            Ballot(ranking=[{"C"}, {"B"}], weight=3),
        ]
    )

    assert remove_cand("A", profile_no_ties) == no_a_true
    assert remove_cand("A", profile_no_ties.get_ballots()) == no_a_true.get_ballots()
    assert remove_cand("A", Ballot(ranking=[{"A"}, {"B"}])) == Ballot(ranking=[{"B"}])


def test_remove_cand_no_ties():
    no_a = remove_cand("A", profile_no_ties)
    no_a_true = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"B"}], weight=1),
            Ballot(ranking=[{"B"}, {"C"}], weight=1 / 2),
            Ballot(ranking=[{"C"}, {"B"}], weight=3),
        ]
    )

    no_a_b = remove_cand(["A", "B"], profile_no_ties)
    no_a_b_true = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"C"}], weight=7 / 2),
        ]
    )

    assert no_a == no_a_true
    assert no_a_b == no_a_b_true


def test_remove_cand_with_ties():
    no_a = remove_cand("A", profile_with_ties)
    no_a_true = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"B"}], weight=1),
            Ballot(ranking=[{"B", "C"}], weight=1 / 2),
            Ballot(ranking=[{"C"}, {"B"}], weight=3),
        ]
    )

    no_a_b = remove_cand(["A", "B"], profile_no_ties)
    no_a_b_true = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"C"}], weight=7 / 2),
        ]
    )
    assert no_a == no_a_true
    assert no_a_b == no_a_b_true


def test_add_missing_cands():
    true_add = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A", "B"}, {"D"}, {"C", "E"}], weight=1),
            Ballot(ranking=[{"A", "B", "C", "D"}, {"E"}], weight=1 / 2),
            Ballot(ranking=[{"A"}, {"C"}, {"B"}, {"D", "E"}], weight=3),
            Ballot(ranking=[{"A"}, {"C"}, {"B"}, {"D"}, {"E"}]),
        ]
    )

    assert add_missing_cands(profile_with_missing) == true_add


def test_validate_score_vector():
    with pytest.raises(ValueError, match="Score vector must be non-negative."):
        validate_score_vector([3, 2, -1])

    with pytest.raises(ValueError, match="Score vector must be non-increasing."):
        validate_score_vector([3, 2, 3])

    validate_score_vector([3, 2, 1, 0])
    validate_score_vector([3, 3, 3, 3])


def test_score_profile():
    true_scores = {
        "A": Fraction(105, 4),
        "B": Fraction(73, 4),
        "C": Fraction(77, 4),
        "D": Fraction(45, 4),
        "E": Fraction(30, 4),
    }

    comp_scores = score_profile(profile_with_missing, [5, 4, 3, 2, 1])
    assert comp_scores == true_scores
    assert isinstance(comp_scores["A"], Fraction)


def test_score_profile_to_float():
    true_scores = {"A": 105 / 4, "B": 73 / 4, "C": 77 / 4, "D": 45 / 4, "E": 30 / 4}

    comp_scores = score_profile(profile_with_missing, [5, 4, 3, 2, 1], to_float=True)
    assert comp_scores == true_scores
    assert isinstance(comp_scores["A"], float)


def test_score_profile_error():
    with pytest.raises(ValueError, match="Score vector must be non-negative."):
        score_profile(PreferenceProfile(), [3, 2, -1])

    with pytest.raises(ValueError, match="Score vector must be non-increasing."):
        score_profile(PreferenceProfile(), [3, 2, 3])


def test_first_place_votes():
    votes = first_place_votes(profile_no_ties)
    true_votes = {"A": Fraction(3, 2), "B": Fraction(0), "C": Fraction(3)}

    assert votes == true_votes
    assert isinstance(votes["A"], Fraction)


def test_first_place_votes_to_float():
    votes = first_place_votes(profile_no_ties, to_float=True)
    true_votes = {"A": 1.5, "B": 0, "C": 3}

    assert votes == true_votes
    assert isinstance(votes["A"], float)


def test_mentions():
    correct = {"A": Fraction(9, 2), "B": Fraction(9, 2), "C": Fraction(7, 2)}
    test = mentions(profile_no_ties)
    assert correct == test
    assert isinstance(test["A"], Fraction)


def test_mentions_to_float():
    correct = {"A": 9 / 2, "B": 9 / 2, "C": 7 / 2}
    test = mentions(profile_no_ties, to_float=True)
    assert correct == test
    assert isinstance(test["A"], float)


def test_borda_no_ties():
    true_borda = {"A": Fraction(15, 2), "B": Fraction(9), "C": Fraction(21, 2)}

    borda = borda_scores(profile_no_ties)

    assert borda == true_borda
    assert isinstance(borda["A"], Fraction)


def test_borda_no_ties_to_float():
    true_borda = {"A": 15 / 2, "B": 9, "C": 21 / 2}

    borda = borda_scores(profile_no_ties, to_float=True)
    assert borda == true_borda
    assert isinstance(borda["A"], float)


def test_borda_with_ties():
    true_borda = {"A": Fraction(25, 2), "B": Fraction(13, 2), "C": Fraction(8)}

    borda = borda_scores(profile_with_ties)

    assert borda == true_borda
    assert isinstance(borda["A"], Fraction)


def test_tie_broken_ranking():
    fpv_ranking = (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    borda_ranking = (frozenset({"A"}), frozenset({"C"}), frozenset({"B"}))
    tied_ranking = (frozenset({"A", "C", "B"}),)
    assert tie_broken_ranking(tied_ranking, profile_with_ties, "none") == tied_ranking
    assert (
        tie_broken_ranking(tied_ranking, profile_with_ties, "firstplace") == fpv_ranking
    )
    assert tie_broken_ranking(tied_ranking, profile_with_ties, "borda") == borda_ranking


def test_score_dict_to_ranking():
    score_dict = {"A": 3, "B": 2, "C": 3, "D": 2, "E": -1, "F": 2.5}
    high_low = score_dict_to_ranking(score_dict)
    low_high = score_dict_to_ranking(score_dict, sort_high_low=False)
    target_order = (
        frozenset({"A", "C"}),
        frozenset({"F"}),
        frozenset({"B", "D"}),
        frozenset({"E"}),
    )
    assert high_low == target_order
    assert low_high == tuple(list(target_order)[::-1])


def test_elect_cands_from_set_ranking():
    elected, remaining = elect_cands_from_set_ranking(
        ({"A", "B"}, {"C"}, {"D", "E"}, {"F"}), 3
    )
    assert elected == ({"A", "B"}, {"C"})
    assert remaining == ({"D", "E"}, {"F"})


def test_elect_cands_from_set_ranking_errors():
    with pytest.raises(ValueError, match="m must be strictly positive"):
        elect_cands_from_set_ranking(({"A", "B"},), 0)

    with pytest.raises(
        ValueError,
        match="Cannot elect correct number of candidates without breaking ties.",
    ):
        elect_cands_from_set_ranking(({"A", "B"},), 1)


def test_expand_tied_ballot():
    ballot = Ballot(ranking=[{"A", "B"}, {"C", "D"}], weight=4)
    no_ties = [
        Ballot(ranking=[{"A"}, {"B"}, {"C"}, {"D"}]),
        Ballot(ranking=[{"B"}, {"A"}, {"C"}, {"D"}]),
        Ballot(ranking=[{"A"}, {"B"}, {"D"}, {"C"}]),
        Ballot(ranking=[{"B"}, {"A"}, {"D"}, {"C"}]),
    ]

    assert set(expand_tied_ballot(ballot)) == set(no_ties)


def test_resolve_profile_ties():
    no_ties = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"B"}], weight=1 / 2),
            Ballot(ranking=[{"B"}, {"A"}], weight=1 / 2),
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=1 / 12),
            Ballot(ranking=[{"B"}, {"C"}, {"A"}], weight=1 / 12),
            Ballot(ranking=[{"B"}, {"A"}, {"C"}], weight=1 / 12),
            Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=1 / 12),
            Ballot(ranking=[{"C"}, {"A"}, {"B"}], weight=1 / 12),
            Ballot(ranking=[{"A"}, {"C"}, {"B"}], weight=37 / 12),
        ]
    )

    assert resolve_profile_ties(profile_with_ties) == no_ties
