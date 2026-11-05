import itertools as it
import math

import numpy as np
import pytest

from votekit.ballot_generator import (
    BlocSlateConfig,
    name_bt_profile_generator,
    name_bt_profile_generator_using_mcmc,
    name_bt_profiles_by_bloc_generator,
)
from votekit.ballot_generator.bloc_slate_generator import name_bradley_terry as name_bt
from votekit.pref_interval import PreferenceInterval, combine_preference_intervals
from votekit.pref_profile import RankProfile


@pytest.mark.parametrize(
    "generator",
    [name_bt_profile_generator, name_bt_profile_generator_using_mcmc],
)
def test_name_bt_retains_grouped_result_and_can_disable_grouping(generator):
    config = BlocSlateConfig(
        n_voters=20,
        slate_to_candidates={"slate": ["A", "B"]},
        bloc_proportions={"bloc": 1.0},
        preference_mapping={"bloc": {"slate": PreferenceInterval({"A": 0.6, "B": 0.4})}},
        cohesion_mapping={"bloc": {"slate": 1.0}},
    )

    ungrouped = generator(config, group_ballots=False, rng_seed=42)
    grouped = generator(config, group_ballots=True, rng_seed=42)

    assert ungrouped.num_ballots == 20
    assert grouped.num_ballots <= 2
    assert grouped.total_ballot_wt == ungrouped.total_ballot_wt == 20
    assert grouped.candidates == ungrouped.candidates
    assert grouped.ballots == ungrouped.group_ballots().ballots


@pytest.mark.parametrize("voters_per_bloc", [7, name_bt.BATCH_SIZE + 3])
@pytest.mark.parametrize("rng_seed", [42, -42])
def test_dp_name_bt_preserves_seeded_rankings_across_batches_and_blocs(voters_per_bloc, rng_seed):
    config = BlocSlateConfig(
        n_voters=2 * voters_per_bloc,
        slate_to_candidates={"slate": ["A", "B"]},
        bloc_proportions={"X": 0.5, "Y": 0.5},
        preference_mapping={
            "X": {"slate": PreferenceInterval({"A": 0.6, "B": 0.4})},
            "Y": {"slate": PreferenceInterval({"A": 0.2, "B": 0.8})},
        },
        cohesion_mapping={"X": {"slate": 1.0}, "Y": {"slate": 1.0}},
    )

    rng = np.random.default_rng(abs(rng_seed))
    expected_rankings = np.empty((config.n_voters, 2), dtype=int)
    for bloc_index, probability_a_first in enumerate((0.6, 0.2)):
        start = bloc_index * voters_per_bloc
        stop = start + voters_per_bloc
        first_candidates = (rng.random(voters_per_bloc) >= probability_a_first).astype(int)
        expected_rankings[start:stop, 0] = first_candidates
        expected_rankings[start:stop, 1] = 1 - first_candidates
        rng.random(voters_per_bloc)  # A draw is consumed even for the forced final candidate.

    profile = name_bt_profile_generator(config, group_ballots=False, rng_seed=rng_seed)

    np.testing.assert_array_equal(profile._matrix.rankings, expected_rankings)
    assert profile.total_ballot_wt == config.n_voters
    assert profile.candidates == ("A", "B")


@pytest.mark.parametrize(
    "n_candidates, n_voters, available_gib",
    [(24, 100_000, 3.0), (2, 50_000_000, 1.0)],
)
def test_dp_memory_check_includes_preparation_and_profile_copies(
    monkeypatch, n_candidates, n_voters, available_gib
):
    monkeypatch.setattr(name_bt, "system_memory", lambda: {"available_gib": available_gib})

    with pytest.raises(MemoryError, match="Estimated peak"):
        name_bt._check_dp_name_bt_memory(
            n_candidates, n_voters, n_positive_candidates=n_candidates, group_ballots=False
        )


def test_dp_memory_check_accounts_for_grouping(monkeypatch):
    monkeypatch.setattr(name_bt, "system_memory", lambda: {"available_gib": 0.4})

    name_bt._check_dp_name_bt_memory(8, 1_000_000, n_positive_candidates=8, group_ballots=False)
    with pytest.raises(MemoryError, match="Not enough memory for DP name-BradleyTerry"):
        name_bt._check_dp_name_bt_memory(8, 1_000_000, n_positive_candidates=8, group_ballots=True)


def test_dp_memory_check_rejects_unsupported_mask_width():
    with pytest.raises(ValueError, match="uint32 subset masks"):
        name_bt._check_dp_name_bt_memory(33, 10, n_positive_candidates=33, group_ballots=False)


@pytest.mark.parametrize(
    "generator", [name_bt_profile_generator, name_bt_profiles_by_bloc_generator]
)
def test_dp_generator_checks_memory_before_building_cdf(monkeypatch, generator):
    config = BlocSlateConfig(
        n_voters=10,
        slate_to_candidates={"slate": ["A", "B"]},
        bloc_proportions={"bloc": 1.0},
        preference_mapping={"bloc": {"slate": PreferenceInterval({"A": 0.6, "B": 0.4})}},
        cohesion_mapping={"bloc": {"slate": 1.0}},
    )
    monkeypatch.setattr(name_bt, "system_memory", lambda: {"available_gib": 0.0})
    monkeypatch.setattr(
        name_bt,
        "_build_cdf_matrix",
        lambda *args: pytest.fail("CDF allocation must not begin when memory is insufficient."),
    )

    with pytest.raises(MemoryError, match="only 0.000 GiB is available"):
        generator(config, rng_seed=42)


@pytest.mark.parametrize("group_ballots", [False, True])
def test_name_bt_combined_and_bloc_profiles_share_samples(group_ballots):
    config = BlocSlateConfig(
        n_voters=35,
        slate_to_candidates={"slate": ["A", 2, "C"]},
        bloc_proportions={"X": 0.6, "Y": 0.4},
        preference_mapping={
            "X": {"slate": PreferenceInterval({"A": 0.6, 2: 0.3, "C": 0.1})},
            "Y": {"slate": PreferenceInterval({"A": 0.1, 2: 0.2, "C": 0.7})},
        },
        cohesion_mapping={"X": {"slate": 1.0}, "Y": {"slate": 1.0}},
    )

    profile = name_bt_profile_generator(config, group_ballots=group_ballots, rng_seed=42)
    by_bloc = name_bt_profiles_by_bloc_generator(config, group_ballots=group_ballots, rng_seed=42)

    assert list(by_bloc) == ["X", "Y"]
    assert [part.total_ballot_wt for part in by_bloc.values()] == [21, 14]
    assert all(part.candidates == profile.candidates == ("A", 2, "C") for part in by_bloc.values())
    combined_ballots = tuple(ballot for part in by_bloc.values() for ballot in part.ballots)
    if group_ballots:
        assert profile == RankProfile(ballots=combined_ballots).group_ballots()
    else:
        assert profile.ballots == combined_ballots


@pytest.mark.parametrize("supports", [(3.0, 2.0, 1.0), (3.0, 0.0, 1.0)])
def test_name_bt_cdf_matches_enumerated_pairwise_probabilities(supports):
    candidates = ("A", "B", "C")
    rankings = list(it.permutations(candidates))
    support_by_candidate = dict(zip(candidates, supports, strict=True))
    weights = {
        ranking: math.prod(
            support_by_candidate[preferred]
            / (support_by_candidate[preferred] + support_by_candidate[other])
            for preferred, other in it.combinations(ranking, 2)
        )
        for ranking in rankings
    }
    normalizer = sum(weights.values())
    with np.errstate(all="raise"):
        cdf = name_bt._build_cdf_matrix(candidates, np.array(supports))

    for ranking in rankings:
        remaining = 7
        probability = 1.0
        for candidate in ranking:
            candidate_id = candidates.index(candidate)
            lower_bound = cdf[remaining, candidate_id - 1] if candidate_id else 0.0
            probability *= cdf[remaining, candidate_id] - lower_bound
            remaining -= 1 << candidate_id
        assert probability == pytest.approx(weights[ranking] / normalizer, abs=1e-14)


@pytest.mark.parametrize(
    "generator", [name_bt_profile_generator, name_bt_profiles_by_bloc_generator]
)
def test_name_bt_ranks_a_zero_support_candidate_last(generator):
    config = BlocSlateConfig(
        n_voters=50,
        slate_to_candidates={"slate": ["A", "B", "C"]},
        bloc_proportions={"bloc": 1.0},
        preference_mapping={
            "bloc": {
                "slate": PreferenceInterval({"A": 0.6, "B": 0.0, "C": 0.4}, allow_zero_support=True)
            }
        },
        cohesion_mapping={"bloc": {"slate": 1.0}},
        allow_zero_support_candidates=True,
    )
    result = generator(config, rng_seed=42)
    profile = result["bloc"] if isinstance(result, dict) else result

    assert profile.total_ballot_wt == 50
    assert all(ballot.ranking[-1] == frozenset({"B"}) for ballot in profile.ballots)


def test_name_bt_randomizes_zero_support_tails_before_grouping(monkeypatch):
    monkeypatch.setattr(name_bt, "BATCH_SIZE", 37)
    config = BlocSlateConfig(
        n_voters=12_000,
        slate_to_candidates={"slate": ["A", "B", "C", "D"]},
        bloc_proportions={"bloc": 1.0},
        preference_mapping={
            "bloc": {
                "slate": PreferenceInterval(
                    {"A": 0.0, "B": 0.6, "C": 0.0, "D": 0.4}, allow_zero_support=True
                )
            }
        },
        cohesion_mapping={"bloc": {"slate": 1.0}},
        allow_zero_support_candidates=True,
    )

    ungrouped = name_bt_profile_generator(config, group_ballots=False, rng_seed=42)
    grouped = name_bt_profile_generator(config, group_ballots=True, rng_seed=42)
    repeated = name_bt_profile_generator(config, group_ballots=False, rng_seed=42)

    np.testing.assert_array_equal(ungrouped._matrix.rankings, repeated._matrix.rankings)
    assert grouped.ballots == ungrouped.group_ballots().ballots
    assert grouped.candidates == ("A", "B", "C", "D")
    assert grouped.total_ballot_wt == config.n_voters
    rankings, counts = np.unique(ungrouped._matrix.rankings, axis=0, return_counts=True)
    probabilities = {
        tuple(ranking): count / config.n_voters
        for ranking, count in zip(rankings, counts, strict=True)
    }
    assert probabilities == pytest.approx(
        {
            (1, 3, 0, 2): 0.3,
            (1, 3, 2, 0): 0.3,
            (3, 1, 0, 2): 0.2,
            (3, 1, 2, 0): 0.2,
        },
        abs=0.02,
    )

    config.allow_zero_support_candidates = False
    with pytest.raises(ValueError, match="strictly positive"):
        name_bt_profile_generator(config, rng_seed=42)


def test_name_bt_zero_support_sets_are_specific_to_each_bloc():
    config = BlocSlateConfig(
        n_voters=200,
        slate_to_candidates={"slate": ["A", "B", "C", "D"]},
        bloc_proportions={"X": 0.5, "Y": 0.5},
        preference_mapping={
            "X": {
                "slate": PreferenceInterval(
                    {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0}, allow_zero_support=True
                )
            },
            "Y": {
                "slate": PreferenceInterval(
                    {"A": 0.0, "B": 0.0, "C": 1.0, "D": 0.0}, allow_zero_support=True
                )
            },
        },
        cohesion_mapping={"X": {"slate": 1.0}, "Y": {"slate": 1.0}},
        allow_zero_support_candidates=True,
    )
    by_bloc = name_bt_profiles_by_bloc_generator(config, group_ballots=False, rng_seed=42)
    grouped = name_bt_profiles_by_bloc_generator(config, group_ballots=True, rng_seed=42)
    combined = name_bt_profile_generator(config, group_ballots=False, rng_seed=42)

    for bloc, positive_id in (("X", 0), ("Y", 2)):
        rankings = by_bloc[bloc]._matrix.rankings
        assert np.all(rankings[:, 0] == positive_id)
        zero_ids = set(range(4)) - {positive_id}
        assert {tuple(row) for row in rankings[:, 1:]} == set(it.permutations(sorted(zero_ids)))
        assert grouped[bloc].ballots == by_bloc[bloc].group_ballots().ballots
        assert grouped[bloc].total_ballot_wt == 100
    np.testing.assert_array_equal(
        combined._matrix.rankings,
        np.concatenate([part._matrix.rankings for part in by_bloc.values()]),
    )


def test_name_bt_zero_support_candidates_do_not_expand_the_dp_table(monkeypatch):
    monkeypatch.setattr(name_bt, "system_memory", lambda: {"available_gib": 0.001})
    candidates = list(range(40))
    config = BlocSlateConfig(
        n_voters=5,
        slate_to_candidates={"slate": candidates},
        bloc_proportions={"bloc": 1.0},
        preference_mapping={
            "bloc": {
                "slate": PreferenceInterval(
                    {candidate: float(candidate in (7, 30)) for candidate in candidates},
                    allow_zero_support=True,
                )
            }
        },
        cohesion_mapping={"bloc": {"slate": 1.0}},
        allow_zero_support_candidates=True,
    )
    profile = name_bt_profile_generator(config, group_ballots=False, rng_seed=42)

    np.testing.assert_array_equal(
        np.sort(profile._matrix.rankings[:, :2], axis=1), np.tile([7, 30], (5, 1))
    )
    np.testing.assert_array_equal(
        np.sort(profile._matrix.rankings, axis=1), np.tile(candidates, (5, 1))
    )
    assert profile.total_ballot_wt == 5


@pytest.mark.parametrize("n_candidates", [1, 13])
def test_name_bt_uses_available_memory_instead_of_the_enumeration_cutoff(n_candidates):
    candidates = list(range(n_candidates))
    config = BlocSlateConfig(
        n_voters=3,
        slate_to_candidates={"slate": candidates},
        bloc_proportions={"bloc": 1.0},
        preference_mapping={
            "bloc": {"slate": PreferenceInterval({candidate: 1.0 for candidate in candidates})}
        },
        cohesion_mapping={"bloc": {"slate": 1.0}},
    )

    profile = name_bt_profile_generator(config, group_ballots=False, rng_seed=42)

    assert profile.total_ballot_wt == 3
    np.testing.assert_array_equal(
        np.sort(profile._matrix.rankings, axis=1), np.tile(candidates, (3, 1))
    )


def test_NBT_completion():
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        bloc_proportions={"W": 0.7, "C": 0.3},
        preference_mapping={
            "W": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "C": {
                "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        cohesion_mapping={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
    )

    profile = name_bt_profile_generator(config)
    assert type(profile) is RankProfile

    profile_dict = name_bt_profiles_by_bloc_generator(config)
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is RankProfile


def test_NBT_completion_with_mixed_candidates():
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates={"W": ["W1", "W2"], "C": [1, 2]},
        bloc_proportions={"W": 0.7, "C": 0.3},
        preference_mapping={
            "W": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({1: 0.2, 2: 0.1}),
            },
            "C": {
                "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "C": PreferenceInterval({1: 0.3, 2: 0.3}),
            },
        },
        cohesion_mapping={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
    )

    profile = name_bt_profile_generator(config)
    assert type(profile) is RankProfile

    profile_dict = name_bt_profiles_by_bloc_generator(config)
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is RankProfile


def test_NBT_distribution(do_ballot_probs_match_ballot_dist_rank_profile):
    # Set-up
    number_of_ballots = 500

    candidates = ["W1", "W2", "C1", "C2"]
    pref_intervals_by_bloc = {
        "X": {
            "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
            "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
        },
        "Y": {
            "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
        },
    }
    bloc_voter_prop = {"X": 0.7, "Y": 0.3}
    cohesion_parameters = {"X": {"W": 0.7, "C": 0.3}, "Y": {"C": 0.6, "W": 0.4}}

    # Generate ballots
    config = BlocSlateConfig(
        n_voters=number_of_ballots,
        slate_to_candidates={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        bloc_proportions=bloc_voter_prop,
        preference_mapping=pref_intervals_by_bloc,
        cohesion_mapping=cohesion_parameters,
    )

    generated_profile = name_bt_profile_generator(config)

    # Find ballot probs
    possible_rankings = list(it.permutations(candidates, len(candidates)))

    final_ballot_prob_dict = {b: 0.0 for b in possible_rankings}
    pref_interval_by_bloc = {
        bloc: combine_preference_intervals(
            [pref_intervals_by_bloc[bloc][slate] for slate in ["W", "C"]],
            [cohesion_parameters[bloc][slate] for slate in ["W", "C"]],
        )
        for bloc in ["X", "Y"]
    }

    for bloc in bloc_voter_prop.keys():
        ballot_prob_dict = {b: 0.0 for b in possible_rankings}
        for ranking in possible_rankings:
            support_for_cands = pref_interval_by_bloc[bloc].interval
            prob = bloc_voter_prop[bloc]
            for i in range(len(ranking)):
                greater_cand = support_for_cands[ranking[i]]
                for j in range(i + 1, len(ranking)):
                    cand = support_for_cands[ranking[j]]
                    prob *= greater_cand / (greater_cand + cand)
            ballot_prob_dict[ranking] += prob
        normalizer = 1 / sum(ballot_prob_dict.values())
        ballot_prob_dict = {k: v * normalizer for k, v in ballot_prob_dict.items()}
        final_ballot_prob_dict = {
            k: v + bloc_voter_prop[bloc] * ballot_prob_dict[k]
            for k, v in final_ballot_prob_dict.items()
        }

    assert isinstance(generated_profile, RankProfile)
    # Test
    assert do_ballot_probs_match_ballot_dist_rank_profile(final_ballot_prob_dict, generated_profile)


def test_NBT_3_bloc(do_ballot_probs_match_ballot_dist_rank_profile):
    slate_to_candidates = {"A": ["A1"], "B": ["B1"], "C": ["C1"]}

    cohesion_parameters = {
        "X": {"A": 0.7, "B": 0.2, "C": 0.1},
        "Y": {"A": 0.7, "B": 0.2, "C": 0.1},
        "Z": {"A": 0.7, "B": 0.2, "C": 0.1},
    }

    pref_intervals_by_bloc = {
        "X": {
            "A": PreferenceInterval({"A1": 1}),
            "B": PreferenceInterval({"B1": 1}),
            "C": PreferenceInterval({"C1": 1}),
        },
        "Y": {
            "A": PreferenceInterval({"A1": 1}),
            "B": PreferenceInterval({"B1": 1}),
            "C": PreferenceInterval({"C1": 1}),
        },
        "Z": {
            "A": PreferenceInterval({"A1": 1}),
            "B": PreferenceInterval({"B1": 1}),
            "C": PreferenceInterval({"C1": 1}),
        },
    }

    bloc_voter_prop = {"X": 0.9998, "Y": 0.0001, "Z": 0.0001}

    # Generate ballots
    config = BlocSlateConfig(
        n_voters=500,
        slate_to_candidates=slate_to_candidates,
        bloc_proportions=bloc_voter_prop,
        preference_mapping=pref_intervals_by_bloc,
        cohesion_mapping=cohesion_parameters,
        silent=True,
    )

    profile = name_bt_profile_generator(config)

    summ = 98 + 28 + 7 + 49 + 4 + 2

    ballot_prob_dict = {
        ("A1", "B1", "C1"): 98 / summ,
        ("A1", "C1", "B1"): 49 / summ,
        ("B1", "A1", "C1"): 28 / summ,
        ("B1", "C1", "A1"): 4 / summ,
        ("C1", "A1", "B1"): 7 / summ,
        ("C1", "B1", "A1"): 2 / summ,
    }

    assert isinstance(profile, RankProfile)

    assert do_ballot_probs_match_ballot_dist_rank_profile(ballot_prob_dict, profile)

    alphas = {
        "X": {"A": 1, "B": 1, "C": 1},
        "Y": {"A": 1, "B": 1, "C": 1},
        "Z": {"A": 1, "B": 1, "C": 1},
    }

    config.set_dirichlet_alphas(alphas)

    profile = name_bt_profile_generator(config)
    assert isinstance(profile, RankProfile)


# FIX: Get this test working
# def test_NBT_MCMC_subsample_distribution(): # Set-up number_of_ballots = 500
#
#     candidates = ["W1", "W2", "C1", "C2"]
#     pref_intervals_by_bloc = {
#         "W": {
#             "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
#             "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
#         },
#         "C": {
#             "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
#             "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
#         },
#     }
#     bloc_voter_prop = {"W": 0.7, "C": 0.3}
#     cohesion_parameters = {"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.6, "W": 0.4}}
#     bloc_voter_prop = {"W": 0.7, "C": 0.3}
#
#     # Generate ballots
#     bt = name_BradleyTerry(
#         candidates=candidates,
#         pref_intervals_by_bloc=pref_intervals_by_bloc,
#         bloc_voter_prop=bloc_voter_prop,
#         cohesion_parameters=cohesion_parameters,
#     )
#     generated_profile = bt.generate_profile_MCMC_even_subsample(
#         number_of_ballots=number_of_ballots
#     )
#
#     # Length of ballot should be number_of_ballots, not chain length
#
#     # chain length < number_of_ballots --> resorts to number_of_ballots
#
#     # Continuous sampling should do worse than spaced out subsampling w.h.p.
#
#     # Acceptance ratio should be roughly be between two values

# FIX: Get this method up and running again
# def test_MCMC_subsample_chain_length_error(): """ Test to check chain_length < num_ballots to be
# sampled """
#
#     # Initialization of parameters; doesn't matter for this test
#     candidates = ["W1", "W2", "C1", "C2"]
#     pref_intervals_by_bloc = {
#         "W": {
#             "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
#             "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
#         },
#         "C": {
#             "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
#             "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
#         },
#     }
#     bloc_voter_prop = {"W": 0.7, "C": 0.3}
#     cohesion_parameters = {"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.6, "W": 0.4}}
#     bloc_voter_prop = {"W": 0.7, "C": 0.3}
#
#     bt = name_BradleyTerry(
#         candidates=candidates,
#         pref_intervals_by_bloc=pref_intervals_by_bloc,
#         bloc_voter_prop=bloc_voter_prop,
#         cohesion_parameters=cohesion_parameters,
#     )
#
#     number_of_ballots = 3001
#
#     # Error where minority bloc needs 0.3*10,000 = 3,000 ballots, where 10,000 is the preset
#     # chain_length.
#     with pytest.raises(
#         ValueError,
#         match=(
#             "The number of ballots to be sampled is more than the chain length; "
#             "supply a greater chain length."
#         ),
#     ):
#         bt.generate_profile_MCMC_even_subsample(number_of_ballots=number_of_ballots)
#
#     chain_length = 100
#     number_of_ballots = 101
#
#     with pytest.raises(
#         ValueError,
#         match=(
#             "The number of ballots to be sampled is more than the chain length; "
#             "supply a greater chain length."
#         ),
#     ):
#         bt.generate_profile_MCMC_even_subsample(
#             number_of_ballots=number_of_ballots, chain_length=chain_length
#         )


def test_name_bt_memory_estimate_bounds_preparation_buffers(monkeypatch):
    monkeypatch.setattr(name_bt, "system_memory", lambda: {"available_gib": 4.0})
    name_bt._check_dp_name_bt_memory(24, 100_000, n_positive_candidates=24, group_ballots=False)


@pytest.mark.parametrize("batch_size", [1, 7])
@pytest.mark.parametrize("supports", [(3.0, 2.0, 1.0, 0.0), (1e-300, 1e-200, 0.1, 1.0)])
def test_name_bt_preparation_batches_preserve_cdf(monkeypatch, batch_size, supports):
    candidates = tuple(range(len(supports)))
    expected = name_bt._build_cdf_matrix(candidates, np.array(supports))
    monkeypatch.setattr(name_bt, "BATCH_SIZE", batch_size)
    actual = name_bt._build_cdf_matrix(candidates, np.array(supports))
    np.testing.assert_array_equal(actual, expected)
