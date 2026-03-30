import itertools as it
import math

import pytest

from votekit.ballot_generator import ic_profile_generator
from votekit.ballot_generator.std_generator import impartial_culture as ic_module
from votekit.pref_profile import RankProfile


def test_IC_completion():
    profile = ic_profile_generator(candidates=["W1", "W2", "C1", "C2"], number_of_ballots=100)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100


def test_ic_distribution(do_ballot_probs_match_ballot_dist_rank_profile):
    # Set-up
    number_of_ballots = 100

    candidates = ["W1", "W2", "C1", "C2"]

    # Find ballot probs
    possible_rankings = it.permutations(candidates, len(candidates))
    ballot_prob_dict = {b: 1 / math.factorial(len(candidates)) for b in possible_rankings}

    # Generate ballots
    generated_profile = ic_profile_generator(
        candidates=candidates, number_of_ballots=number_of_ballots
    )

    assert isinstance(generated_profile, RankProfile)
    # Test
    assert do_ballot_probs_match_ballot_dist_rank_profile(ballot_prob_dict, generated_profile)


def test_ic_non_short_helper_defaults_max_ballot_length(monkeypatch):
    def fake_choice(num_cands, size, replace):
        assert num_cands == 3
        assert size == 3
        assert replace is False
        return [2, 0, 1]

    monkeypatch.setattr(ic_module.np.random, "choice", fake_choice)

    profile = ic_module._generate_profile_optimized_non_short(
        candidates=["A", "B", "C"],
        number_of_ballots=2,
    )

    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 2
    assert profile.max_ranking_length == 3
    assert len(profile.ballots) == 1
    assert profile.ballots[0].weight == 2
    assert tuple(
        next(iter(rank))
        for rank in profile.ballots[0].ranking  # ty: ignore[not-iterable]
    ) == (
        "C",
        "A",
        "B",
    )


def test_ic_allow_short_ballots_uses_lexicographic_indices(monkeypatch):
    ballot_inds = iter([0, 1, 8])

    def fake_randint(low, high):
        assert low == 0
        assert high == 8
        return next(ballot_inds)

    monkeypatch.setattr(ic_module.random, "randint", fake_randint)

    profile = ic_profile_generator(
        candidates=["A", "B", "C"],
        number_of_ballots=3,
        max_ballot_length=2,
        allow_short_ballots=True,
    )

    ballot_weights = {
        tuple(
            next(iter(rank))
            for rank in ballot.ranking  # ty: ignore[not-iterable]
        ): ballot.weight
        for ballot in profile.ballots
    }

    assert ballot_weights == {("A",): 1, ("A", "B"): 1, ("C", "B"): 1}
    assert profile.max_ranking_length == 3


def test_ic_short_helper_defaults_max_ballot_length(monkeypatch):
    ballot_inds = iter([0, 14])

    def fake_randint(low, high):
        assert low == 0
        assert high == 14
        return next(ballot_inds)

    monkeypatch.setattr(ic_module.random, "randint", fake_randint)

    profile = ic_module._generate_profile_optimized_with_short(
        candidates=["A", "B", "C"],
        number_of_ballots=2,
    )

    assert profile.ballots is not None
    ballot_weights = {
        tuple(
            next(iter(rank))
            for rank in ballot.ranking  # ty: ignore[not-iterable]
        ): ballot.weight
        for ballot in profile.ballots
    }

    assert ballot_weights == {("A",): 1, ("C", "B", "A"): 1}
    assert profile.max_ranking_length == 3


def test_ic_rejects_max_ballot_length_larger_than_number_of_candidates():
    with pytest.raises(
        ValueError, match="Max ballot length larger than number of candidates given."
    ):
        ic_profile_generator(
            candidates=["A", "B", "C"],
            number_of_ballots=5,
            max_ballot_length=4,
        )
