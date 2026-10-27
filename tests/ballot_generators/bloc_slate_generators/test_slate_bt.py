import itertools as it
from collections import Counter
from fractions import Fraction
from typing import cast

import numpy as np
import pytest

from votekit.ballot import RankBallot
from votekit.ballot_generator import (
    BlocSlateConfig,
    slate_bt_profile_generator,
    slate_bt_profiles_by_bloc_generator,
)
from votekit.ballot_generator.bloc_slate_generator import slate_bradley_terry as slate_bt
from votekit.pref_interval import PreferenceInterval
from votekit.pref_profile import RankProfile
from votekit.types import Candidate

PROB_THRESHOLD = 0.01


def ballot_ranking(ballot: RankBallot) -> tuple[frozenset[Candidate], ...]:
    assert ballot.ranking is not None
    return ballot.ranking


def compute_sbt_slate_ballot_distribution(config: BlocSlateConfig, bloc: str):
    """
    Compute the probability distribution for ballot types for a given voter bloc.

    Args:
        config (BlocSlateConfig): The configuration for the bloc-slate type model.
        bloc (str): The voter bloc for which to compute the ballot type distribution.

    Returns:
        dict[tuple[str, ...], float]: A dictionary mapping ballot types (as tuples of
            slate names) to their probabilities.
    """
    slates_with_multiplicity = [
        slate for slate in config.slates for _ in range(len(config.slate_to_candidates[slate]))
    ]
    slate_ballot_dist = {
        slate_ballot_type: 1.0 for slate_ballot_type in it.permutations(slates_with_multiplicity)
    }
    for slate_ballot_type in slate_ballot_dist.keys():
        for idx, slate in enumerate(slate_ballot_type):
            for other_slate in slate_ballot_type[idx + 1 :]:
                if slate != other_slate:
                    cohesion = cast(float, config.cohesion_df.at[bloc, slate])
                    other_cohesion = cast(float, config.cohesion_df.at[bloc, other_slate])
                    slate_ballot_dist[slate_ballot_type] *= cohesion / (cohesion + other_cohesion)

    return {
        slate_ballot_type: mass / sum(slate_ballot_dist.values())
        for slate_ballot_type, mass in slate_ballot_dist.items()
    }


def test_SBT_completion(two_bloc_two_slate_config):
    config = two_bloc_two_slate_config
    profile = slate_bt_profile_generator(config)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100_000

    profile_dict = slate_bt_profiles_by_bloc_generator(config)
    assert isinstance(profile_dict, dict)
    assert type(profile_dict["X"]) is RankProfile
    assert type(profile_dict["Y"]) is RankProfile


def test_SBT_completion_with_mixed_candidates(two_bloc_two_slate_mixed_config):
    config = two_bloc_two_slate_mixed_config
    profile = slate_bt_profile_generator(config)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100_000

    profile_dict = slate_bt_profiles_by_bloc_generator(config)
    assert isinstance(profile_dict, dict)
    assert type(profile_dict["X"]) is RankProfile
    assert type(profile_dict["Y"]) is RankProfile


def test_SBT_invalid_config():
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates={
            "Z": ["A1", "A2"],
            "B": ["B1", "B2"],
        },  # invalid slate name
        bloc_proportions={"X": 0.7, "Y": 0.3},
        preference_mapping={
            "X": {
                "A": PreferenceInterval({"A1": 0.4, "A2": 0.3}),
                "B": PreferenceInterval({"B1": 0.2, "B2": 0.1}),
            },
            "Y": {
                "A": PreferenceInterval({"A1": 0.2, "A2": 0.2}),
                "B": PreferenceInterval({"B1": 0.3, "B2": 0.3}),
            },
        },
        cohesion_mapping={"X": {"A": 0.7, "B": 0.3}, "Y": {"B": 0.9, "A": 0.1}},
        silent=True,
    )
    with pytest.raises(
        KeyError,
        match=(
            r"cohesion_df columns \(slates\) must be exactly \['Z', 'B'\] as defined in the "
            r"'slate_to_candidates' parameter. Got \['A', 'B'\]"
        ),
    ):
        slate_bt_profile_generator(config)


def test_slate_bt_count_dp_accepts_twenty_four_candidates():
    n = 8
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates={
            "A": [f"A{i}" for i in range(n)],
            "B": [f"B{i}" for i in range(n)],
            "C": [f"C{i}" for i in range(n)],
        },
        bloc_proportions={"X": 1},
        preference_mapping={
            "X": {
                "A": PreferenceInterval({f"A{i}": 1 for i in range(n)}),
                "B": PreferenceInterval({f"B{i}": 1 for i in range(n)}),
                "C": PreferenceInterval({f"C{i}": 1 for i in range(n)}),
            },
        },
        cohesion_mapping={"X": {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}},
    )

    profile = slate_bt_profile_generator(config, group_ballots=False, rng_seed=42)

    assert profile.num_ballots == 100
    assert profile.total_ballot_wt == 100
    assert profile.candidates == tuple(config.candidates)
    np.testing.assert_array_equal(
        np.sort(profile._matrix.rankings, axis=1), np.tile(np.arange(24), (100, 1))
    )


def test_SBT_two_bloc_two_slate_distribution_matches_slate_ballot_dist(
    two_bloc_two_slate_config,
):
    config = two_bloc_two_slate_config
    profile_dict = slate_bt_profiles_by_bloc_generator(config)

    for bloc in config.blocs:
        profile = profile_dict[bloc]
        slate_ballot_dist = compute_sbt_slate_ballot_distribution(config, bloc)
        cand_to_slate_dict = {
            cand: slate
            for slate, cand_list in config.slate_to_candidates.items()
            for cand in cand_list
        }
        slate_ballot_counts = {}

        for ballot in profile.ballots:
            ranking = ballot_ranking(ballot)
            if any(len(cand_set) > 1 for cand_set in ranking):
                raise ValueError(f"Tie occurred in ballot {ranking}")
            slate_ballot_type = tuple(
                [cand_to_slate_dict[cand] for cand_set in ranking for cand in cand_set]
            )
            slate_ballot_counts[slate_ballot_type] = (
                slate_ballot_counts.get(slate_ballot_type, 0) + ballot.weight
            )

        assert all(
            abs(
                slate_ballot_weight / profile.total_ballot_wt - slate_ballot_dist[slate_ballot_type]
            )
            < PROB_THRESHOLD
            for slate_ballot_type, slate_ballot_weight in slate_ballot_counts.items()
        )


def test_two_bloc_two_slate_sbt_distribution_matches_name_ballot_dist(
    two_bloc_two_slate_config,
):
    config = two_bloc_two_slate_config

    profiles_by_bloc = slate_bt_profiles_by_bloc_generator(config, group_ballots=True)

    for bloc in config.blocs:
        profile = profiles_by_bloc[bloc]

        a_comparisons_profile = [
            tuple(
                cand
                for cand_set in ballot_ranking(ballot)
                for cand in cand_set
                if isinstance(cand, str) and cand[0] == "A"
            )
            for ballot in profile.ballots
            for _ in range(int(ballot.weight))
        ]

        b_comparisons_profile = [
            tuple(
                cand
                for cand_set in ballot_ranking(ballot)
                for cand in cand_set
                if isinstance(cand, str) and cand[0] == "B"
            )
            for ballot in profile.ballots
            for _ in range(int(ballot.weight))
        ]

        assert (
            abs(
                a_comparisons_profile.count(("A1", "A2")) / profile.total_ballot_wt
                - config.preference_df.loc[bloc]["A1"]
                / (config.preference_df.loc[bloc]["A1"] + config.preference_df.loc[bloc]["A2"])
            )
            < PROB_THRESHOLD
        )

        assert (
            abs(
                b_comparisons_profile.count(("B1", "B2")) / profile.total_ballot_wt
                - config.preference_df.loc[bloc]["B1"]
                / (config.preference_df.loc[bloc]["B1"] + config.preference_df.loc[bloc]["B2"])
            )
            < PROB_THRESHOLD
        )


def test_SBT_one_bloc_three_slate_distribution_matches_slate_ballot_dist(
    one_bloc_three_slate_config,
):
    config = one_bloc_three_slate_config
    profile = slate_bt_profile_generator(config)

    bloc = config.blocs[0]
    slate_ballot_dist = compute_sbt_slate_ballot_distribution(config, bloc)
    cand_to_slate_dict = {
        cand: slate for slate, cand_list in config.slate_to_candidates.items() for cand in cand_list
    }
    slate_ballot_counts = {}

    for ballot in profile.ballots:
        ranking = ballot_ranking(ballot)
        if any(len(cand_set) > 1 for cand_set in ranking):
            raise ValueError(f"Tie occurred in ballot {ranking}")
        slate_ballot_type = tuple(
            [cand_to_slate_dict[cand] for cand_set in ranking for cand in cand_set]
        )
        slate_ballot_counts[slate_ballot_type] = (
            slate_ballot_counts.get(slate_ballot_type, 0) + ballot.weight
        )

    assert all(
        abs(slate_ballot_weight / profile.total_ballot_wt - slate_ballot_dist[slate_ballot_type])
        < PROB_THRESHOLD
        for slate_ballot_type, slate_ballot_weight in slate_ballot_counts.items()
    )


def test_one_bloc_three_slate_sbt_distribution_matches_name_ballot_dist(
    one_bloc_three_slate_config,
):
    config = one_bloc_three_slate_config

    profile = slate_bt_profile_generator(config, group_ballots=True)

    for slate in config.slates:
        cand_comparisons_profile = [
            tuple(
                cand
                for cand_set in ballot_ranking(ballot)
                for cand in cand_set
                if isinstance(cand, str) and cand[0] == slate and cand[-1] in ["1", "2"]
            )
            for ballot in profile.ballots
            for _ in range(int(ballot.weight))
        ]

        assert (
            abs(
                cand_comparisons_profile.count((f"{slate}1", f"{slate}2")) / profile.total_ballot_wt
                - config.preference_df.loc["X"][f"{slate}1"]
                / (
                    config.preference_df.loc["X"][f"{slate}1"]
                    + config.preference_df.loc["X"][f"{slate}2"]
                )
            )
            < PROB_THRESHOLD
        )


def test_sbt_zero_support_slates():
    config = BlocSlateConfig(
        n_voters=100_000,
        allow_zero_support_candidates=True,
        slate_to_candidates={"A": ["A1", "A2"], "B": ["B1", "B2"], "C": ["C1"]},
        bloc_proportions={"X": 1},
        preference_mapping={
            "X": {
                "A": PreferenceInterval({"A1": 0.4, "A2": 0.3}),
                "B": PreferenceInterval({"B1": 0.2, "B2": 0.1}),
                "C": PreferenceInterval({"C1": 1}),
            },
        },
        cohesion_mapping={"X": {"A": 1, "B": 0, "C": 0}},
    )

    profile = slate_bt_profile_generator(config)
    zero_support_slate_perms = [
        tuple(
            cand[0]
            for cand_set in ballot_ranking(ballot)
            for cand in cand_set
            if isinstance(cand, str) and cand[0] != "A"
        )
        for ballot in profile.ballots
        for _ in range(int(ballot.weight))
    ]
    zero_support_slate_perms_dist = Counter(zero_support_slate_perms)
    zero_support_slate_perms_dist = {
        b: w / sum(zero_support_slate_perms_dist.values())
        for b, w in zero_support_slate_perms_dist.items()
    }

    assert all(
        abs(prob - 1 / 3) < PROB_THRESHOLD for prob in zero_support_slate_perms_dist.values()
    )


@pytest.mark.parametrize("counts", [(3,), (2, 1), (2, 1, 1), (1, 2, 2), (1, 1, 1, 1)])
def test_slate_dp_probabilities_match_enumerated_pairwise_weights(counts):
    supports = [Fraction(index + 1) for index in range(len(counts))]
    symbols = [slate for slate, count in enumerate(counts) for _ in range(count)]
    weights = {}
    for sequence in set(it.permutations(symbols)):
        weight = Fraction(1)
        for position, slate in enumerate(sequence):
            for other in sequence[position + 1 :]:
                if slate != other:
                    weight *= supports[slate] / (supports[slate] + supports[other])
        weights[sequence] = weight
    normalizer = sum(weights.values())

    cdf, strides = slate_bt._build_cdf_matrix(
        np.array(counts, dtype=int), np.array(supports, dtype=np.float64)
    )
    for sequence, weight in weights.items():
        state = len(cdf) - 1
        probability = 1.0
        for slate in sequence:
            lower_threshold = cdf[state, slate - 1] if slate else 0.0
            probability *= cdf[state, slate] - lower_threshold
            state -= strides[slate]
        assert state == 0
        assert probability == pytest.approx(float(weight / normalizer), rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("scale", [1.0, 1e-250, 1e250])
def test_slate_dp_cdfs_preserve_scaling_and_exhausted_slates(scale):
    counts = np.array([2, 3, 1], dtype=int)
    supports = np.array([3.0, 2.0, 1.0])
    expected, _ = slate_bt._build_cdf_matrix(counts, supports)
    cdf, strides = slate_bt._build_cdf_matrix(counts, scale * supports)

    np.testing.assert_allclose(cdf, expected, rtol=1e-11, atol=1e-12)
    assert np.all(np.isfinite(cdf))
    assert np.all(np.diff(cdf, axis=1) >= 0)
    np.testing.assert_array_equal(cdf[1:, -1], 1.0)
    probabilities = np.diff(cdf, axis=1, prepend=0.0)
    for slate, stride in enumerate(strides):
        exhausted = (np.arange(len(cdf)) // stride) % (counts[slate] + 1) == 0
        np.testing.assert_array_equal(probabilities[exhausted, slate], 0.0)


def test_slate_sequence_sampling_preserves_counts_and_batch_draw_order(monkeypatch):
    counts = np.array([2, 1, 3, 1], dtype=int)
    cdf, strides = slate_bt._build_cdf_matrix(counts, np.array([0.1, 0.2, 0.3, 0.4]))
    expected = np.empty((29, int(counts.sum())), dtype=int)
    actual = np.empty_like(expected)
    first_rng = np.random.default_rng(42)
    second_rng = np.random.default_rng(42)
    slate_bt._sample_sequences_from_cdf(cdf, strides, expected, first_rng)
    monkeypatch.setattr(slate_bt, "_DP_SLATE_BT_BATCH_SIZE", 4)
    slate_bt._sample_sequences_from_cdf(cdf, strides, actual, second_rng)

    np.testing.assert_array_equal(actual, expected)
    assert first_rng.random() == second_rng.random()
    for slate, count in enumerate(counts):
        np.testing.assert_array_equal((actual == slate).sum(axis=1), count)


def test_slate_sequence_sampling_skips_exhausted_columns_at_zero_draw():
    class ZeroDraws:
        def random(self, size):
            return np.zeros(size)

    counts = np.array([1, 2, 1], dtype=int)
    cdf, strides = slate_bt._build_cdf_matrix(counts, np.ones(3))
    actual = np.empty((1, 4), dtype=int)
    slate_bt._sample_sequences_from_cdf(
        cdf, strides, actual, cast(np.random.Generator, ZeroDraws())
    )

    np.testing.assert_array_equal(actual, [[0, 1, 1, 2]])


def test_slate_bt_public_apis_share_matrix_samples_and_grouping(
    monkeypatch, two_bloc_two_slate_mixed_config
):
    config = two_bloc_two_slate_mixed_config
    config.n_voters = 37

    def reject_dataframe_construction(*args, **kwargs):
        raise AssertionError("Direct slate BT must construct its matrix without a dataframe.")

    monkeypatch.setattr(RankProfile, "__init__", reject_dataframe_construction)
    monkeypatch.setattr(RankProfile, "to_df", reject_dataframe_construction)
    combined = slate_bt_profile_generator(config, group_ballots=False, rng_seed=42)
    by_bloc = slate_bt_profiles_by_bloc_generator(config, group_ballots=False, rng_seed=42)
    grouped = slate_bt_profile_generator(config, group_ballots=True, rng_seed=42)
    grouped_by_bloc = slate_bt_profiles_by_bloc_generator(config, group_ballots=True, rng_seed=42)

    np.testing.assert_array_equal(
        combined._matrix.rankings,
        np.concatenate([profile._matrix.rankings for profile in by_bloc.values()]),
    )
    assert combined.num_ballots == combined.total_ballot_wt == config.n_voters
    assert combined.candidates == tuple(config.candidates)
    assert grouped.ballots == combined.group_ballots().ballots
    assert grouped.candidates == combined.candidates
    for bloc, profile in by_bloc.items():
        assert profile.candidates == combined.candidates
        assert grouped_by_bloc[bloc].ballots == profile.group_ballots().ballots


def test_slate_bt_preserves_zero_cohesion_and_zero_candidate_tails():
    config = BlocSlateConfig(
        n_voters=12000,
        allow_zero_support_candidates=True,
        slate_to_candidates={"A": [10, "a0", "a1"], "B": [20, "b0", "b1"], "C": [30]},
        bloc_proportions={"bloc": 1.0},
        preference_mapping={
            "bloc": {
                "A": PreferenceInterval({10: 1.0, "a0": 0.0, "a1": 0.0}, allow_zero_support=True),
                "B": PreferenceInterval({20: 1.0, "b0": 0.0, "b1": 0.0}, allow_zero_support=True),
                "C": PreferenceInterval({30: 1.0}),
            }
        },
        cohesion_mapping={"bloc": {"A": 1.0, "B": 0.0, "C": 0.0}},
    )
    profile = slate_bt_profile_generator(config, group_ballots=False, rng_seed=4)
    rankings = profile._matrix.rankings
    np.testing.assert_array_equal(np.sort(rankings, axis=1), np.tile(np.arange(7), (12000, 1)))
    np.testing.assert_array_equal(rankings[:, 0], 0)
    np.testing.assert_array_equal(np.sort(rankings[:, 1:3], axis=1), np.tile([1, 2], (12000, 1)))
    assert np.mean(rankings[:, 1] == 1) == pytest.approx(0.5, abs=0.02)

    b_orderings = rankings[(rankings >= 3) & (rankings <= 5)].reshape(-1, 3)
    np.testing.assert_array_equal(b_orderings[:, 0], 3)
    assert np.mean(b_orderings[:, 1] == 4) == pytest.approx(0.5, abs=0.02)
    for position in range(3, 7):
        assert np.mean(rankings[:, position] == 6) == pytest.approx(0.25, abs=0.02)


def test_slate_bt_single_candidate():
    config = BlocSlateConfig(
        n_voters=10,
        slate_to_candidates={"A": [7]},
        bloc_proportions={"bloc": 1.0},
        preference_mapping={"bloc": {"A": PreferenceInterval({7: 1.0})}},
        cohesion_mapping={"bloc": {"A": 1.0}},
    )
    by_bloc = slate_bt_profiles_by_bloc_generator(config, rng_seed=42)
    assert by_bloc["bloc"].num_ballots == 1
    assert by_bloc["bloc"].total_ballot_wt == 10
    assert by_bloc["bloc"].candidates == (7,)
    np.testing.assert_array_equal(by_bloc["bloc"]._matrix.rankings, [[0]])


def test_slate_bt_memory_rejection_precedes_table_allocation(
    monkeypatch, two_bloc_two_slate_config
):
    monkeypatch.setattr(slate_bt, "system_memory", lambda: {"available_gib": 0.0})

    def reject_table_allocation(*args, **kwargs):
        raise AssertionError("Memory check must run before table allocation.")

    monkeypatch.setattr(slate_bt, "_build_cdf_matrix", reject_table_allocation)
    with pytest.raises(MemoryError, match="Estimated peak"):
        slate_bt_profile_generator(two_bloc_two_slate_config)


def test_slate_bt_memory_estimate_includes_output_and_grouping(monkeypatch):
    monkeypatch.setattr(slate_bt, "system_memory", lambda: {"available_gib": 0.4})
    counts = np.array([4, 4], dtype=int)
    cohesion = np.array([[0.5, 0.5]])

    slate_bt._check_dp_slate_bt_memory(counts, cohesion, 1_000_000, group_ballots=False)
    with pytest.raises(MemoryError, match="Estimated peak"):
        slate_bt._check_dp_slate_bt_memory(counts, cohesion, 1_000_000, group_ballots=True)
    with pytest.raises(MemoryError, match="Estimated peak"):
        slate_bt._check_dp_slate_bt_memory(counts, cohesion, 50_000_000, group_ballots=False)


def test_slate_bt_memory_estimate_uses_positive_cohesion_counts(monkeypatch):
    monkeypatch.setattr(slate_bt, "system_memory", lambda: {"available_gib": 0.01})
    counts = np.full(8, 8, dtype=int)
    slate_bt._check_dp_slate_bt_memory(
        counts, np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]]), 1, group_ballots=False
    )
    with pytest.raises(MemoryError, match="Estimated peak"):
        slate_bt._check_dp_slate_bt_memory(counts, np.full((1, 8), 1 / 8), 1, group_ballots=False)


def test_slate_bt_rejects_count_table_index_overflow():
    with pytest.raises(ValueError, match="array-index width"):
        slate_bt._check_dp_slate_bt_memory(
            np.ones(64, dtype=int), np.full((1, 64), 1 / 64), 1, group_ballots=False
        )


def test_slate_bt_within_slate_order_is_plackett_luce():
    supports: dict[Candidate, float] = {"A": 1.0, "B": 2.0, "C": 3.0}
    config = BlocSlateConfig(
        n_voters=30000,
        slate_to_candidates={"slate": list(supports)},
        bloc_proportions={"bloc": 1.0},
        preference_mapping={"bloc": {"slate": PreferenceInterval(supports)}},
        cohesion_mapping={"bloc": {"slate": 1.0}},
    )
    profile = slate_bt_profile_generator(config, rng_seed=13)
    assert profile.num_ballots == 6
    for ballot in profile.ballots:
        remaining_support = sum(supports.values())
        expected_probability = 1.0
        for position in ballot_ranking(ballot):
            candidate = next(iter(position))
            support = supports[candidate]
            expected_probability *= support / remaining_support
            remaining_support -= support
        assert ballot.weight / config.n_voters == pytest.approx(expected_probability, abs=0.01)
