from votekit.ballot_generator import (
    name_bt_profile_generator,
    name_bt_profiles_by_bloc_generator,
    BlocSlateConfig,
)
from votekit.pref_profile import RankProfile
from votekit.pref_interval import PreferenceInterval, combine_preference_intervals
from votekit.ballot_generator.bloc_slate_generator.name_bradley_terry import (
    _calc_prob as bt_prob,
)
import itertools as it


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
    assert do_ballot_probs_match_ballot_dist_rank_profile(
        final_ballot_prob_dict, generated_profile
    )


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


def test_NBT_probability_calculation():
    # Set-up
    slate_to_candidate = {"W": ["W1", "W2"], "C": ["C1", "C2"]}
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

    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates=slate_to_candidate,
        bloc_proportions=bloc_voter_prop,
        preference_mapping=pref_intervals_by_bloc,
        cohesion_mapping=cohesion_parameters,
        silent=True,
    )

    permutation = ("W1", "W2")

    pref_interval_by_bloc = config.get_combined_preference_intervals_by_bloc()
    x_pref_interval = pref_interval_by_bloc["X"].interval
    y_pref_interval = pref_interval_by_bloc["Y"].interval

    assert bt_prob(permutations=[permutation], cand_support_dict=dict(y_pref_interval))[
        permutation
    ] == (y_pref_interval["W1"] / (y_pref_interval["W1"] + y_pref_interval["W2"]))

    permutation = ("W1", "W2", "C2")
    prob = (
        (x_pref_interval["W1"] / (x_pref_interval["W1"] + x_pref_interval["W2"]))
        * (x_pref_interval["W1"] / (x_pref_interval["W1"] + x_pref_interval["C2"]))
        * (x_pref_interval["W2"] / (x_pref_interval["W2"] + x_pref_interval["C2"]))
    )
    assert (
        bt_prob(permutations=[permutation], cand_support_dict=dict(x_pref_interval))[
            permutation
        ]
        == prob
    )


# FIX: Get this test working
# def test_NBT_MCMC_subsample_distribution():
#     # Set-up
#     number_of_ballots = 500
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
# def test_MCMC_subsample_chain_length_error():
#     """
#     Test to check chain_length < num_ballots to be sampled
#     """
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
#     # Error where minority bloc needs 0.3*10,000 = 3,000 number of ballots, where 10,000 is the preset chain_length
#     with pytest.raises(
#         ValueError,
#         match="The number of ballots to be sampled is more than the chain length; supply a greater chain length.",
#     ):
#         bt.generate_profile_MCMC_even_subsample(number_of_ballots=number_of_ballots)
#
#     chain_length = 100
#     number_of_ballots = 101
#
#     with pytest.raises(
#         ValueError,
#         match="The number of ballots to be sampled is more than the chain length; supply a greater chain length.",
#     ):
#         bt.generate_profile_MCMC_even_subsample(
#             number_of_ballots=number_of_ballots, chain_length=chain_length
#         )
