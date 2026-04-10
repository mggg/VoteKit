import math
import random
from collections import Counter
from itertools import combinations

import pytest

from votekit.ballot_generator import iac_profile_generator
from votekit.ballot_generator.std_generator import impartial_anon_culture as iac_module
from votekit.pref_profile import RankProfile


def _ballot_counter(profile: RankProfile) -> Counter[tuple[str, ...]]:
    return Counter(
        {
            tuple(next(iter(rank)) for rank in ballot.ranking): ballot.weight
            for ballot in profile.ballots
            if ballot.ranking is not None
        }
    )


def _single_ballot_count_pmf(num_ballot_types: int, number_of_ballots: int) -> list[float]:
    total_profiles = math.comb(number_of_ballots + num_ballot_types - 1, num_ballot_types - 1)
    return [
        math.comb(number_of_ballots - count + num_ballot_types - 2, num_ballot_types - 2)
        / total_profiles
        for count in range(number_of_ballots + 1)
    ]


def _sample_ballot_count_cdf(
    *,
    max_ballot_length: int,
    number_of_draws: int,
    tracked_ballot_index: int,
) -> list[float]:
    state = random.getstate()
    random.seed(0)

    try:
        counts = Counter()
        for _ in range(number_of_draws):
            ballot_counts = iac_module._sample_anonymous_profile_ballot_counts(
                n_candidates=3,
                number_of_ballots=10,
                max_ballot_length=max_ballot_length,
            )
            counts[ballot_counts[tracked_ballot_index]] += 1
    finally:
        random.setstate(state)

    running = 0
    cdf = []
    for count in range(11):
        running += counts[count]
        cdf.append(running / number_of_draws)
    return cdf


def test_IAC_completion():
    profile = iac_profile_generator(candidates=["W1", "W2", "C1", "C2"], number_of_ballots=100)

    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100


def test_iac_maps_stars_and_bars_sample_to_lexicographic_ballots(monkeypatch):
    def fake_sample(population, k):
        assert list(population) == list(range(5))
        assert k == 3
        return [0, 2, 4]

    monkeypatch.setattr(iac_module.random, "sample", fake_sample)

    profile = iac_profile_generator(candidates=["A", "B"], number_of_ballots=2)

    assert _ballot_counter(profile) == Counter({("A", "B"): 1, ("B",): 1})
    assert profile.max_ranking_length == 2


def test_iac_respects_max_ballot_length(monkeypatch):
    def fake_sample(population, k):
        assert list(population) == list(range(4))
        assert k == 2
        return [1, 2]

    monkeypatch.setattr(iac_module.random, "sample", fake_sample)

    profile = iac_profile_generator(
        candidates=["A", "B", "C"],
        number_of_ballots=2,
        max_ballot_length=1,
    )

    assert _ballot_counter(profile) == Counter({("A",): 1, ("C",): 1})
    assert all(len(ballot) == 1 for ballot in _ballot_counter(profile))


def test_iac_profile_count_sampler_is_uniform_for_small_case():
    state = random.getstate()
    random.seed(0)

    try:
        draws = Counter(tuple(iac_module._sample_uniform_profile_counts(4, 2)) for _ in range(5000))
    finally:
        random.setstate(state)

    expected_num_profiles = math.comb(2 + 4 - 1, 4 - 1)
    expected_count = 5000 / expected_num_profiles

    assert len(draws) == expected_num_profiles
    assert all(abs(count - expected_count) < 120 for count in draws.values())


def test_iac_profile_count_sampler_rejects_nonpositive_num_ballot_types():
    with pytest.raises(ValueError, match="num_ballot_types must be positive"):
        iac_module._sample_uniform_profile_counts(0, 2)


def test_iac_profile_count_sampler_handles_single_ballot_type():
    assert iac_module._sample_uniform_profile_counts(1, 7) == [7]


def test_iac_raises_for_invalid_max_ballot_length():
    with pytest.raises(
        ValueError, match="Max ballot length larger than number of candidates given."
    ):
        iac_profile_generator(
            candidates=["A", "B", "C"],
            number_of_ballots=5,
            max_ballot_length=4,
        )


@pytest.mark.parametrize(
    ("max_ballot_length", "num_ballot_types"),
    [(3, 15), (2, 9)],
)
def test_iac_distribution_for_three_candidates_and_ten_voters(
    max_ballot_length: int, num_ballot_types: int
):
    number_of_draws = 50000

    empirical_cdf = _sample_ballot_count_cdf(
        max_ballot_length=max_ballot_length,
        number_of_draws=number_of_draws,
        tracked_ballot_index=0,
    )
    pmf = _single_ballot_count_pmf(num_ballot_types, 10)

    expected_cdf = []
    running = 0.0
    for probability in pmf:
        running += probability
        expected_cdf.append(running)

    max_cdf_error = max(
        abs(observed - expected) for observed, expected in zip(empirical_cdf, expected_cdf)
    )

    assert max_cdf_error < 0.015


def test_iac_rejects_wrong_number_of_bar_locations():
    with pytest.raises(
        ValueError,
        match="bar_locations must contain exactly num_ballot_types - 1 entries",
    ):
        iac_module._bar_locations_to_profile_counts(
            bar_locations=[0],
            num_ballot_types=3,
            number_of_ballots=2,
        )


@pytest.mark.parametrize(
    ("max_ballot_length", "num_ballot_types"),
    [(3, 15), (2, 9)],
)
def test_iac_enumerates_every_anonymous_profile_for_three_candidates_and_ten_voters(
    max_ballot_length: int, num_ballot_types: int
):
    number_of_ballots = 10
    total_slots = number_of_ballots + num_ballot_types - 1
    expected_num_profiles = math.comb(total_slots, num_ballot_types - 1)

    observed_profiles = 0
    for bar_locations in combinations(range(total_slots), num_ballot_types - 1):
        profile_counts = iac_module._bar_locations_to_profile_counts(
            bar_locations=bar_locations,
            num_ballot_types=num_ballot_types,
            number_of_ballots=number_of_ballots,
        )

        assert len(profile_counts) == num_ballot_types
        assert sum(profile_counts) == number_of_ballots
        assert all(count >= 0 for count in profile_counts)
        assert iac_module._profile_counts_to_bar_locations(profile_counts) == list(bar_locations)

        observed_profiles += 1

    assert observed_profiles == expected_num_profiles
