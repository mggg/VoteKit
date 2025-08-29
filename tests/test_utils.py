from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile, ProfileError
from votekit.utils import (
    ballots_by_first_cand,
    add_missing_cands,
    validate_score_vector,
    score_profile_from_rankings,
    first_place_votes,
    mentions,
    borda_scores,
    tiebreak_set,
    tiebroken_ranking,
    score_dict_to_ranking,
    elect_cands_from_set_ranking,
    expand_tied_ballot,
    resolve_profile_ties,
    score_profile_from_ballot_scores,
    ballot_lengths,
)
import pytest

profile_no_ties = PreferenceProfile(
    ballots=(
        Ballot(ranking=tuple(map(frozenset, [{"A"}, {"B"}])), weight=1),
        Ballot(ranking=tuple(map(frozenset, [{"A"}, {"B"}, {"C"}])), weight=1 / 2),
        Ballot(ranking=tuple(map(frozenset, [{"C"}, {"B"}, {"A"}])), weight=3),
    )
)

profile_with_ties = PreferenceProfile(
    ballots=(
        Ballot(ranking=tuple(map(frozenset, [{"A", "B"}])), weight=1),
        Ballot(ranking=tuple(map(frozenset, [{"A", "B", "C"}])), weight=1 / 2),
        Ballot(ranking=tuple(map(frozenset, [{"A"}, {"C"}, {"B"}])), weight=3),
    )
)

profile_with_missing = PreferenceProfile(
    ballots=(
        Ballot(ranking=tuple(map(frozenset, [{"A", "B"}, {"D"}])), weight=1),
        Ballot(ranking=tuple(map(frozenset, [{"A", "B", "C", "D"}])), weight=1 / 2),
        Ballot(ranking=tuple(map(frozenset, [{"A"}, {"C"}, {"B"}])), weight=3),
        Ballot(ranking=tuple(map(frozenset, [{"A"}, {"C"}, {"B"}, {"D"}, {"E"}]))),
    ),
    candidates=("A", "B", "C", "D", "E"),
)


def test_ballots_by_first_cand():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=tuple(map(frozenset, [{"A"}, {"B"}])), weight=1),
            Ballot(ranking=tuple(map(frozenset, [{"A"}, {"B"}, {"C"}])), weight=1 / 2),
            Ballot(ranking=tuple(map(frozenset, [{"C"}, {"B"}, {"A"}])), weight=3),
            Ballot(scores={"C": 1}),
        )
    )
    cand_dict = ballots_by_first_cand(profile)
    partition = {
        "A": [
            Ballot(ranking=tuple(map(frozenset, [{"A"}, {"B"}])), weight=1),
            Ballot(ranking=tuple(map(frozenset, [{"A"}, {"B"}, {"C"}])), weight=1 / 2),
        ],
        "B": [],
        "C": [Ballot(ranking=tuple(map(frozenset, [{"C"}, {"B"}, {"A"}])), weight=3)],
    }

    assert cand_dict == partition


def test_ballots_by_first_cand_error():
    with pytest.raises(ValueError, match="has a tie for first."):
        ballots_by_first_cand(profile_with_ties)

    with pytest.raises(TypeError, match="Ballots must have rankings."):
        ballots_by_first_cand(PreferenceProfile(ballots=(Ballot(scores={"A": 3}),)))


def test_add_missing_cands():
    true_add = PreferenceProfile(
        ballots=(
            Ballot(
                ranking=tuple(map(frozenset, [{"A", "B"}, {"D"}, {"C", "E"}])), weight=1
            ),
            Ballot(
                ranking=tuple(map(frozenset, [{"A", "B", "C", "D"}, {"E"}])),
                weight=1 / 2,
            ),
            Ballot(
                ranking=tuple(map(frozenset, [{"A"}, {"C"}, {"B"}, {"D", "E"}])),
                weight=3,
            ),
            Ballot(ranking=tuple(map(frozenset, [{"A"}, {"C"}, {"B"}, {"D"}, {"E"}]))),
        )
    )

    assert add_missing_cands(profile_with_missing) == true_add


def test_add_missing_cands_errors():
    with pytest.raises(TypeError, match="Ballots must have rankings."):
        add_missing_cands(
            PreferenceProfile(ballots=(Ballot(scores={"A": 3}),), candidates=("A", "B"))
        )


def test_validate_score_vector():
    with pytest.raises(ValueError, match="Score vector must be non-negative."):
        validate_score_vector([3, 2, -1])

    with pytest.raises(ValueError, match="Score vector must be non-increasing."):
        validate_score_vector([3, 2, 3])

    validate_score_vector([3, 2, 1, 0])
    validate_score_vector([3, 3, 3, 3])


def test_score_profile_from_rankings_low():
    true_scores = {
        "A": 25,
        "B": 17,
        "C": 17,
        "D": 6,
        "E": 1,
    }

    comp_scores = score_profile_from_rankings(profile_with_missing, [5, 4, 3, 2, 1])
    assert comp_scores == true_scores
    assert isinstance(comp_scores["A"], float)


def test_score_profile_from_rankings_high():
    true_scores = {
        "A": 27.5,
        "B": 19.5,
        "C": 18.5,
        "D": 7.5,
        "E": 1,
    }

    comp_scores = score_profile_from_rankings(
        profile_with_missing, [5, 4, 3, 2, 1], tie_convention="high"
    )
    assert comp_scores == true_scores
    assert isinstance(comp_scores["A"], float)


def test_score_profile_from_rankings_avg():
    true_scores = {
        "A": 26.25,
        "B": 18.25,
        "C": 17.75,
        "D": 6.75,
        "E": 1,
    }

    comp_scores = score_profile_from_rankings(
        profile_with_missing, [5, 4, 3, 2, 1], tie_convention="average"
    )
    assert comp_scores == true_scores
    assert isinstance(comp_scores["A"], float)


def test_score_profile_from_rankings_errors():
    with pytest.raises(ValueError, match="Score vector must be non-negative."):
        score_profile_from_rankings(PreferenceProfile(), [3, 2, -1])

    with pytest.raises(ValueError, match="Score vector must be non-increasing."):
        score_profile_from_rankings(PreferenceProfile(), [3, 2, 3])

    with pytest.raises(ProfileError, match="Profile must only contain ranked ballots."):
        score_profile_from_rankings(
            PreferenceProfile(ballots=(Ballot(scores={"A": 3}),)), [3, 2, 1]
        )

    with pytest.raises(TypeError, match="has an empty ranking position."):
        score_profile_from_rankings(
            PreferenceProfile(ballots=(Ballot(ranking=({"A"}, frozenset(), {"B"})),)),
            [3, 2, 1],
        )
    with pytest.raises(
        ValueError,
        match=("tie_convention must be one of 'high', 'low', 'average', " "not highlo"),
    ):
        score_profile_from_rankings(
            profile_no_ties, [5, 4, 3, 2, 1], tie_convention="highlo"  # type: ignore[arg-type]
        )


def test_first_place_votes():
    votes = first_place_votes(profile_no_ties)
    true_votes = {"A": 3 / 2, "B": 0, "C": 3}

    assert votes == true_votes
    assert isinstance(votes["A"], float)


def test_fpv_errors():
    with pytest.raises(ProfileError, match="Profile must only contain ranked ballots."):
        first_place_votes(PreferenceProfile(ballots=(Ballot(scores={"A": 3}),)))


def test_mentions():
    correct = {"A": 9 / 2, "B": 9 / 2, "C": 7 / 2}
    test = mentions(profile_no_ties)
    assert correct == test
    assert isinstance(test["A"], float)


def test_mentions_errors():
    with pytest.raises(TypeError, match="Ballots must have rankings."):
        mentions(PreferenceProfile(ballots=(Ballot(scores={"A": 3}),)))


def test_borda_no_ties():
    true_borda = {"A": 15 / 2, "B": 9, "C": 19 / 2}

    borda = borda_scores(profile_no_ties)

    assert borda == true_borda
    assert isinstance(borda["A"], float)


def test_borda_with_ties():
    true_borda = {"A": 25 / 2, "B": 13 / 2, "C": 7}

    borda = borda_scores(profile_with_ties, tie_convention="average")

    assert borda == true_borda
    assert isinstance(borda["A"], float)


def test_borda_short_ballot():
    true_borda = {
        "A": 2,
        "B": 2,
        "C": 2,
    }

    borda = borda_scores(
        PreferenceProfile(
            ballots=(
                Ballot(ranking=(frozenset({"A"}), frozenset({"B"}))),
                Ballot(ranking=(frozenset({"C"}), frozenset({"B"}))),
            ),
            candidates=("A", "B", "C", "D"),
        )
    )

    assert borda == true_borda

    true_borda = {
        "A": 1,
        "B": 0,
        "C": 1,
    }

    borda = borda_scores(
        PreferenceProfile(
            ballots=(Ballot(ranking=({"A"}, {"B"})), Ballot(ranking=({"C"}, {"B"}))),
            candidates=["A", "B", "C", "D"],
        ),
        borda_max=1,
    )

    assert borda == true_borda


def test_borda_mismatched_length():
    true_borda = {
        "A": 50,
        "B": 98,
        "C": 50,
    }

    borda = borda_scores(
        PreferenceProfile(
            ballots=(Ballot(ranking=({"A"}, {"B"})), Ballot(ranking=({"C"}, {"B"}))),
            candidates=["A", "B", "C", "D"],
        ),
        borda_max=50,
    )

    assert borda == true_borda

    true_borda = {
        "A": 1,
        "B": 0,
        "C": 1,
    }

    borda = borda_scores(
        PreferenceProfile(
            ballots=(
                Ballot(ranking=({"A"}, {"B"}, {"C"})),
                Ballot(ranking=({"C"}, {"B"})),
            ),
            candidates=["A", "B", "C", "D"],
        ),
        borda_max=1,
    )

    assert borda == true_borda


def test_borda_errors():
    with pytest.raises(ProfileError, match="Profile must only contain ranked ballots."):
        borda_scores(PreferenceProfile(ballots=(Ballot(scores={"A": 3}),)))


def test_tiebreak_set():
    fpv_ranking = (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    borda_ranking = (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    lex_ranking = (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    tied_set = frozenset({"A", "C", "B"})

    profile = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"B"}], weight=1.5),
            Ballot(ranking=[{"A", "B", "C"}], weight=1 / 2),
            Ballot(ranking=[{"A"}, {"C"}, {"B"}], weight=3),
        ]
    )

    assert tiebreak_set(tied_set, profile, "first_place") == fpv_ranking
    assert tiebreak_set(tied_set, profile, "borda") == borda_ranking
    assert tiebreak_set(tied_set, profile, "lex") == lex_ranking
    assert tiebreak_set(tied_set, profile, "lexicographic") == lex_ranking
    assert tiebreak_set(tied_set, profile, "alph") == lex_ranking
    assert tiebreak_set(tied_set, profile, "alphabetical") == lex_ranking
    assert len(tiebreak_set(tied_set)) == 3


def test_tiebreak_set_errors():
    tied_set = frozenset({"A", "C", "B"})
    with pytest.raises(ValueError, match="Method of tiebreak requires profile."):
        tiebreak_set(tied_set, tiebreak="first_place")
    with pytest.raises(ValueError, match="Method of tiebreak requires profile."):
        tiebreak_set(tied_set, tiebreak="borda")
    with pytest.raises(ValueError, match="Invalid tiebreak code was provided"):
        tiebreak_set(tied_set, profile_no_ties, tiebreak="mine")


def test_tiebreak_no_res():
    profile = PreferenceProfile(
        ballots=[
            Ballot(ranking=({"A"},), weight=2),
            Ballot(ranking=({"B"},), weight=2),
            Ballot(ranking=({"C"},)),
        ]
    )
    assert len(tiebreak_set(frozenset({"A", "B", "C"}), profile, "first_place")) == 3


def test_tiebroken_ranking():
    profile = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"B"}], weight=1.5),
            Ballot(ranking=[{"A", "B", "C"}], weight=1 / 2),
            Ballot(ranking=[{"A"}, {"C"}, {"B"}], weight=3),
        ],
        candidates=["A", "B", "C", "D"],
    )

    fpv_ranking = (
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C"}),
        frozenset({"D"}),
    )
    borda_ranking = (
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C"}),
        frozenset({"D"}),
    )
    tied_ranking = (frozenset({"A", "C", "B"}), frozenset({"D"}))
    assert tiebroken_ranking(tied_ranking, profile, "first_place")[0] == fpv_ranking
    assert tiebroken_ranking(tied_ranking, profile, "borda")[0] == borda_ranking
    assert len(tiebroken_ranking(tied_ranking)[0]) == 4

    assert tiebroken_ranking(tied_ranking, profile, "first_place")[1] == {
        frozenset({"A", "C", "B"}): (
            frozenset({"A"}),
            frozenset({"B"}),
            frozenset({"C"}),
        )
    }


def test_tiebroken_ranking_errors():
    tied_ranking = (frozenset({"A", "C", "B"}),)
    with pytest.raises(ValueError, match="Method of tiebreak requires profile."):
        tiebroken_ranking(tied_ranking, tiebreak="first_place")
    with pytest.raises(ValueError, match="Method of tiebreak requires profile."):
        tiebroken_ranking(tied_ranking, tiebreak="borda")
    with pytest.raises(ValueError, match="Invalid tiebreak code was provided"):
        tiebroken_ranking(tied_ranking, profile_no_ties, tiebreak="mine")


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
    elected, remaining, tiebroken_ranking = elect_cands_from_set_ranking(
        ({"A", "B"}, {"C"}, {"D", "E"}, {"F"}), 3
    )
    assert elected == ({"A", "B"}, {"C"})
    assert remaining == ({"D", "E"}, {"F"})
    assert not tiebroken_ranking


def test_elect_cands_from_set_ranking_tiebreaks():
    ranking = ({"D", "E"}, {"A"}, {"B", "C"}, {"F"})

    profile = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"B"}], weight=1.5),
            Ballot(ranking=[{"A", "B", "C"}], weight=1 / 2),
            Ballot(ranking=[{"A"}, {"C"}, {"B"}], weight=3),
        ],
        candidates=["A", "B", "C", "D", "E", "F"],
    )

    fpv_elected, fpv_remaining, fpv_tiebroken_ranking = elect_cands_from_set_ranking(
        ranking, 4, profile, tiebreak="first_place"
    )
    (
        borda_elected,
        borda_remaining,
        borda_tiebroken_ranking,
    ) = elect_cands_from_set_ranking(ranking, 4, profile, tiebreak="borda")
    (
        random_elected,
        _,
        _,
    ) = elect_cands_from_set_ranking(ranking, 4, tiebreak="random")

    assert fpv_elected == (frozenset({"D", "E"}), frozenset({"A"}), frozenset({"B"}))
    assert fpv_remaining == (frozenset({"C"}), frozenset({"F"}))
    assert fpv_tiebroken_ranking == (
        frozenset({"B", "C"}),
        (frozenset({"B"}), frozenset({"C"})),
    )

    assert borda_elected == (frozenset({"D", "E"}), frozenset({"A"}), frozenset({"B"}))
    assert borda_remaining == (frozenset({"C"}), frozenset({"F"}))
    assert borda_tiebroken_ranking == (
        frozenset({"B", "C"}),
        (frozenset({"B"}), frozenset({"C"})),
    )

    assert len([c for s in random_elected for c in s]) == 4


def test_elect_cands_from_set_ranking_errors():
    with pytest.raises(ValueError, match="m must be strictly positive"):
        elect_cands_from_set_ranking(({"A", "B"},), 0)

    with pytest.raises(
        ValueError, match="m must be no more than the number of candidates."
    ):
        elect_cands_from_set_ranking(({"A", "B"},), 3)

    with pytest.raises(
        ValueError,
        match="Cannot elect correct number of candidates without breaking ties.",
    ):
        elect_cands_from_set_ranking(({"A", "B"},), 1)

    with pytest.raises(ValueError, match="Method of tiebreak requires profile."):
        elect_cands_from_set_ranking(({"A", "B"},), 1, tiebreak="first_place")
    with pytest.raises(ValueError, match="Method of tiebreak requires profile."):
        elect_cands_from_set_ranking(({"A", "B"},), 1, tiebreak="borda")


def test_expand_tied_ballot():
    ballot = Ballot(ranking=tuple(map(frozenset, [{"A", "B"}, {"C", "D"}])), weight=4)
    no_ties = [
        Ballot(ranking=tuple(map(frozenset, [{"A"}, {"B"}, {"C"}, {"D"}]))),
        Ballot(ranking=tuple(map(frozenset, [{"B"}, {"A"}, {"C"}, {"D"}]))),
        Ballot(ranking=tuple(map(frozenset, [{"A"}, {"B"}, {"D"}, {"C"}]))),
        Ballot(ranking=tuple(map(frozenset, [{"B"}, {"A"}, {"D"}, {"C"}]))),
    ]

    assert set(expand_tied_ballot(ballot)) == set(no_ties)


def test_expand_tied_ballot_errors():
    with pytest.raises(TypeError, match="Ballot must have ranking."):
        expand_tied_ballot(Ballot(scores={"A": 3}))


def test_resolve_profile_ties():
    no_ties = PreferenceProfile(
        ballots=[
            Ballot(ranking=tuple(map(frozenset, [{"A"}, {"B"}])), weight=1 / 2),
            Ballot(ranking=tuple(map(frozenset, [{"B"}, {"A"}])), weight=1 / 2),
            Ballot(ranking=tuple(map(frozenset, [{"A"}, {"B"}, {"C"}])), weight=1 / 12),
            Ballot(ranking=tuple(map(frozenset, [{"B"}, {"C"}, {"A"}])), weight=1 / 12),
            Ballot(ranking=tuple(map(frozenset, [{"B"}, {"A"}, {"C"}])), weight=1 / 12),
            Ballot(ranking=tuple(map(frozenset, [{"C"}, {"B"}, {"A"}])), weight=1 / 12),
            Ballot(ranking=tuple(map(frozenset, [{"C"}, {"A"}, {"B"}])), weight=1 / 12),
            Ballot(
                ranking=tuple(map(frozenset, [{"A"}, {"C"}, {"B"}])), weight=37 / 12
            ),
        ]
    )

    assert resolve_profile_ties(profile_with_ties) == no_ties


def test_score_profile_from_ballot_scores():
    profile = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=(frozenset({"A"}),), scores={"A": 2, "B": 0, "C": 4}, weight=2
            ),
            Ballot(
                scores={"A": 3},
            ),
        ]
    )
    scores = score_profile_from_ballot_scores(profile)
    assert scores == {"A": 7, "C": 8}
    assert isinstance(scores["A"], float)


def test_score_profile_from_ballot_scores_float():
    profile = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=(frozenset({"A"}),), scores={"A": 2, "B": 0, "C": 4}, weight=2
            ),
            Ballot(
                scores={"A": 3},
            ),
        ]
    )
    scores = score_profile_from_ballot_scores(profile)
    assert scores == {"A": 7.0, "C": 8.0}
    assert isinstance(scores["A"], float)


def test_score_profile_from_ballot_scores_error():
    profile = PreferenceProfile(
        ballots=(Ballot(ranking=(frozenset({"A"}),), weight=2),)
    )
    with pytest.raises(TypeError, match="has no scores."):
        score_profile_from_ballot_scores(profile)


def test_ballot_lengths():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=tuple(map(frozenset, [{"A"}, {"B"}, {"C"}, {"D"}]))),
            Ballot(ranking=tuple(map(frozenset, [{"B", "A"}, {"C"}, {"D"}]))),
            Ballot(ranking=tuple(map(frozenset, [{"B", "A"}, {"C"}, {"D"}]))),
            Ballot(ranking=tuple(map(frozenset, [{"A"}, {"C"}])), weight=3 / 2),
            Ballot(ranking=tuple(map(frozenset, [{"B"}])), weight=2),
        ),
        max_ranking_length=5,
    )

    assert ballot_lengths(profile) == {1: 2, 2: 3 / 2, 3: 2, 4: 1, 5: 0}


def test_ballot_lengths_ranking_error():
    profile = PreferenceProfile(ballots=(Ballot(scores={"A": 3}),))
    with pytest.raises(TypeError, match="All ballots must have rankings."):
        ballot_lengths(profile)
