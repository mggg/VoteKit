import itertools
from collections import Counter

import numpy as np
import pytest

from votekit.ballot_generator import (
    BlocSlateConfig,
    name_bt_profile_generator,
    name_bt_profiles_by_bloc_generator,
    name_pl_profile_generator,
    name_pl_profiles_by_bloc_generator,
    slate_bt_profile_generator,
    slate_bt_profiles_by_bloc_generator,
    slate_pl_profile_generator,
    slate_pl_profiles_by_bloc_generator,
)
from votekit.ballot_generator.bloc_slate_generator import name_plackett_luce as name_pl
from votekit.ballot_generator.bloc_slate_generator import slate_bradley_terry as slate_bt
from votekit.ballot_generator.bloc_slate_generator import slate_plackett_luce as slate_pl
from votekit.ballot_generator.bloc_slate_generator._profile_construction import (
    _build_profile_from_rankings,
)
from votekit.ballot_generator.bloc_slate_generator._sampling import (
    _fill_pl_rankings,
    _normalize_log_weights_to_cdf_in_place,
)
from votekit.pref_interval import PreferenceInterval
from votekit.pref_profile import RankProfile


def test_cdf_normalization_overwrites_its_view_and_returns_independent_totals():
    storage = np.full((2, 6), -99.0)
    log_weights = storage[:, ::2]
    log_weights[0] = np.log([27.0, 16.0, 5.0])
    log_weights[1] = [-np.inf, np.log(2.0), np.log(3.0)]

    log_totals = _normalize_log_weights_to_cdf_in_place(log_weights)

    np.testing.assert_allclose(log_weights, [[27 / 48, 43 / 48, 1], [0, 2 / 5, 1]])
    np.testing.assert_allclose(log_totals, np.log([48.0, 5.0]))
    np.testing.assert_array_equal(storage[:, 1::2], -99.0)
    assert not np.shares_memory(log_totals, storage)


@pytest.mark.parametrize(
    "combined_generator, bloc_generator",
    [
        (name_bt_profile_generator, name_bt_profiles_by_bloc_generator),
        (name_pl_profile_generator, name_pl_profiles_by_bloc_generator),
        (slate_bt_profile_generator, slate_bt_profiles_by_bloc_generator),
        (slate_pl_profile_generator, slate_pl_profiles_by_bloc_generator),
    ],
)
def test_ranked_generators_construct_matrices_and_preserve_bloc_metadata(
    monkeypatch, combined_generator, bloc_generator
):
    monkeypatch.setattr(name_pl, "BATCH_SIZE", 7)
    monkeypatch.setattr(slate_bt, "BATCH_SIZE", 7)
    monkeypatch.setattr(slate_pl, "BATCH_SIZE", 7)
    config = BlocSlateConfig(
        n_voters=71,
        slate_to_candidates={"A": [7, "a"], "B": [9, "b"]},
        bloc_proportions={"X": 0.6, "Y": 0.399, "small": 0.001},
        preference_mapping={
            bloc: {
                "A": PreferenceInterval({7: 0.7, "a": 0.3}),
                "B": PreferenceInterval({9: 0.4, "b": 0.6}),
            }
            for bloc in ("X", "Y", "small")
        },
        cohesion_mapping={
            "X": {"A": 0.2, "B": 0.8},
            "Y": {"A": 0.7, "B": 0.3},
            "small": {"A": 0.5, "B": 0.5},
        },
    )

    def reject_dataframe_route(*args, **kwargs):
        raise AssertionError("Sampled rankings must construct their matrices directly.")

    monkeypatch.setattr(RankProfile, "__init__", reject_dataframe_route)
    monkeypatch.setattr(RankProfile, "to_df", reject_dataframe_route)
    combined = combined_generator(config, group_ballots=False, rng_seed=42)
    by_bloc = bloc_generator(config, group_ballots=False, rng_seed=42)
    grouped = combined_generator(config, group_ballots=True, rng_seed=42)
    grouped_by_bloc = bloc_generator(config, group_ballots=True, rng_seed=42)

    np.testing.assert_array_equal(
        combined._matrix.rankings,
        np.concatenate([profile._matrix.rankings for profile in by_bloc.values()]),
    )
    assert combined.num_ballots == combined.total_ballot_wt == config.n_voters
    assert combined.candidates == tuple(config.candidates)
    assert grouped.ballots == combined.group_ballots().ballots
    assert grouped.candidates == combined.candidates
    assert list(by_bloc) == ["X", "Y", "small"]

    for bloc, profile in by_bloc.items():
        assert profile.candidates == combined.candidates
        assert profile.max_ranking_length == len(config.candidates)
        assert grouped_by_bloc[bloc].ballots == profile.group_ballots().ballots
        assert grouped_by_bloc[bloc].candidates == combined.candidates

    assert by_bloc["small"].num_ballots == 1
    assert grouped_by_bloc["small"].num_ballots == 1


@pytest.mark.parametrize("generator", [name_pl_profile_generator, slate_pl_profile_generator])
def test_pl_generators_preserve_zero_support_tails_across_batches(generator):
    config = BlocSlateConfig(
        n_voters=65_539,
        allow_zero_support_candidates=True,
        slate_to_candidates={"A": [7, "a", "b"], "B": [9]},
        bloc_proportions={"X": 1.0},
        preference_mapping={
            "X": {
                "A": PreferenceInterval({7: 1.0, "a": 0, "b": 0}, allow_zero_support=True),
                "B": PreferenceInterval({9: 1.0}),
            }
        },
        cohesion_mapping={"X": {"A": 1.0, "B": 0.0}},
    )
    profile = generator(config, group_ballots=False, rng_seed=42)
    rankings = profile._matrix.rankings

    assert profile.num_ballots == profile.total_ballot_wt == config.n_voters
    np.testing.assert_array_equal(np.sort(rankings, axis=1), np.tile(np.arange(4), (65_539, 1)))
    assert np.all(rankings[:, 0] == 0)
    if generator is slate_pl_profile_generator:
        assert np.all(rankings[:, -1] == 3)
        assert np.mean(rankings[:, 1] == 1) == pytest.approx(0.5, abs=0.01)
    else:
        for candidate_id in (1, 2, 3):
            assert np.mean(rankings[:, 1] == candidate_id) == pytest.approx(1 / 3, abs=0.01)


@pytest.mark.parametrize("generator", [name_pl_profile_generator, slate_pl_profile_generator])
def test_pl_generators_match_complete_ranking_probabilities(generator):
    config = BlocSlateConfig(
        n_voters=30_000,
        slate_to_candidates={"A": ["a", "b", "c"]},
        bloc_proportions={"X": 1.0},
        preference_mapping={"X": {"A": PreferenceInterval({"a": 3, "b": 2, "c": 1})}},
        cohesion_mapping={"X": {"A": 1.0}},
    )
    profile = generator(config, group_ballots=False, rng_seed=72)
    observed = Counter(map(tuple, profile._matrix.rankings))
    supports = [3, 2, 1]

    assert len(observed) == 6
    for ranking in itertools.permutations(range(3)):
        probability = 1.0
        remaining_support = sum(supports)
        for candidate in ranking:
            probability *= supports[candidate] / remaining_support
            remaining_support -= supports[candidate]
        assert observed[ranking] / config.n_voters == pytest.approx(probability, abs=0.01)


@pytest.mark.parametrize("group_ballots", [False, True])
@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (5, 0)])
def test_generated_profile_preserves_empty_dimensions(group_ballots, shape):
    rankings = np.empty(shape, dtype=int)
    profile = _build_profile_from_rankings(
        rankings, tuple(range(shape[1])), group_ballots=group_ballots
    )
    expected_rows = min(shape[0], 1) if group_ballots else shape[0]
    assert profile._matrix.rankings.shape == (expected_rows, shape[1])
    assert profile.total_ballot_wt == shape[0]
    assert profile.candidates == tuple(range(shape[1]))


def test_generated_grouping_preserves_first_appearance_counts_and_ownership():
    rankings = np.array([[2, 0, 1], [0, 2, 1], [2, 0, 1], [1, 2, 0], [0, 2, 1]])
    profile = _build_profile_from_rankings(rankings, ("C", 7, "A"), group_ballots=True)
    rankings[:] = -1

    np.testing.assert_array_equal(profile._matrix.rankings, [[2, 0, 1], [0, 2, 1], [1, 2, 0]])
    np.testing.assert_array_equal(profile._matrix.voter_weight_data.weights, [2.0, 2.0, 1.0])
    assert profile.candidates == ("C", 7, "A")
    assert profile.total_ballot_wt == 5
    assert not profile._matrix.rankings.flags.writeable


@pytest.mark.parametrize("supports", [[0.0, 2.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
def test_pl_sampling_preserves_inputs_and_fills_strided_output(supports):
    candidate_ids = np.array([5, 2, 7, 9])
    weights = np.array(supports)
    expected = np.empty((23, 4), dtype=int)
    _fill_pl_rankings(candidate_ids, weights, expected, np.random.default_rng(42))

    storage = np.full((23, 8), -99, dtype=int)
    _fill_pl_rankings(candidate_ids, weights, storage[:, ::2], np.random.default_rng(42))
    np.testing.assert_array_equal(candidate_ids, [5, 2, 7, 9])
    np.testing.assert_array_equal(weights, supports)
    np.testing.assert_array_equal(storage[:, ::2], expected)
    np.testing.assert_array_equal(storage[:, 1::2], -99)
    np.testing.assert_array_equal(np.sort(expected, axis=1), np.tile([2, 5, 7, 9], (23, 1)))
    if supports[1] > 0:
        np.testing.assert_array_equal(np.sort(expected[:, :2], axis=1), np.tile([2, 7], (23, 1)))
