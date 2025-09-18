import itertools as it
import math
import scipy.stats as stats
from pathlib import Path
import pickle
import numpy as np
from collections import Counter

from votekit.ballot_generator import (
    ImpartialAnonymousCulture,
    ImpartialCulture,
    name_PlackettLuce,
    CambridgeSampler,
    slate_PlackettLuce,
    slate_BradleyTerry,
    name_Cumulative,
    sample_cohesion_ballot_types,
    BlocSlateConfig,
)
from votekit.ballot_generator.bloc_slate_generator.name_bradley_terry import (
    generate_name_bt_profile,
    generate_name_bt_profiles_by_bloc,
    generate_name_bt_profile_using_mcmc,
    generate_name_bt_profiles_by_bloc_using_mcmc,
    _calc_prob as bt_prob,
)
from votekit.pref_profile import PreferenceProfile
from votekit.pref_interval import PreferenceInterval, combine_preference_intervals
from votekit import Ballot


# set seed for more consistent tests
np.random.seed(8675309)


def binomial_confidence_interval(probability, n_attempts, alpha=0.95):
    # Calculate the mean and standard deviation of the binomial distribution
    mean = n_attempts * probability
    std_dev = math.sqrt(n_attempts * probability * (1 - probability))

    # Calculate the confidence interval
    z_score = stats.norm.ppf((1 + alpha) / 2)  # Z-score for 99% confidence level
    margin_of_error = z_score * (std_dev)
    conf_interval = (mean - margin_of_error, mean + margin_of_error)

    return conf_interval


def do_ballot_probs_match_ballot_dist(
    ballot_prob_dict: dict, generated_profile: PreferenceProfile, alpha=0.95
):
    n_ballots = generated_profile.total_ballot_wt
    ballot_conf_dict = {
        b: binomial_confidence_interval(p, n_attempts=int(n_ballots), alpha=alpha)
        for b, p in ballot_prob_dict.items()
    }

    failed = 0

    for b in ballot_conf_dict.keys():
        b_list = [{c} for c in b]
        ballot = next(
            (
                element
                for element in generated_profile.ballots
                if element.ranking == b_list
            ),
            None,
        )
        ballot_weight = 0.0
        if ballot is not None:
            ballot_weight = ballot.weight
        if not (
            int(ballot_conf_dict[b][0]) <= ballot_weight <= int(ballot_conf_dict[b][1])
        ):
            failed += 1

    # allow for small margin of error given confidence intereval
    failure_thresold = round((1 - alpha) * n_ballots)
    return failed <= failure_thresold


def test_ic_distribution():
    # Set-up
    number_of_ballots = 100

    candidates = ["W1", "W2", "C1", "C2"]

    # Find ballot probs
    possible_rankings = it.permutations(candidates, len(candidates))
    ballot_prob_dict = {
        b: 1 / math.factorial(len(candidates)) for b in possible_rankings
    }

    # Generate ballots
    generated_profile = ImpartialCulture(
        candidates=candidates,
    ).generate_profile(number_of_ballots=number_of_ballots)

    assert isinstance(generated_profile, PreferenceProfile)
    # Test
    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, generated_profile)


def test_iac_distribution():
    number_of_ballots = 1000
    ballot_length = 4
    candidates = ["W1", "W2", "C1", "C2"]

    # Find ballot probs
    possible_rankings = list(it.permutations(candidates, ballot_length))
    probabilities = np.random.dirichlet([1] * len(possible_rankings))

    ballot_prob_dict = {
        possible_rankings[b_ind]: probabilities[b_ind]
        for b_ind in range(len(possible_rankings))
    }
    generated_profile = ImpartialAnonymousCulture(
        number_of_ballots=number_of_ballots,
        candidates=candidates,
    ).generate_profile(number_of_ballots=500)

    assert isinstance(generated_profile, PreferenceProfile)
    # Test
    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, generated_profile)


"""
# NOTE: Enable this test once the optimized version is completed
def test_iac_optimized_distribution():

    number_of_ballots = 1000
    ballot_length = 4
    candidates = ["W1", "W2", "C1", "C2"]

    # Find ballot probs
    possible_rankings = list(it.permutations(candidates, ballot_length))
    probabilities = np.random.dirichlet([1] * len(possible_rankings))

    ballot_prob_dict = {
        possible_rankings[b_ind]: probabilities[b_ind]
        for b_ind in range(len(possible_rankings))
    }
    iac_inst = ImpartialAnonymousCulture(
        number_of_ballots=number_of_ballots,
        candidates=candidates,
    )
    if not iac_inst._OPTIMIZED_ENABLED:
        # NOTE: no test if performed if the optimized profile
        # generation is not enabled
        assert True
        return

    generated_profile = iac_inst.generate_profile(
        number_of_ballots=500, use_optimized=True
    )
    # Test
    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, generated_profile)
"""


def test_NPL_distribution():
    # Set-up
    number_of_ballots = 100
    candidates = ["W1", "W2", "C1", "C2"]

    pref_intervals_by_bloc = {
        "W": {
            "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
            "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
        },
        "C": {
            "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
        },
    }
    bloc_voter_prop = {"W": 0.7, "C": 0.3}
    cohesion_parameters = {"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.6, "W": 0.4}}

    # Generate ballots
    pl = name_PlackettLuce(
        candidates=candidates,
        pref_intervals_by_bloc=pref_intervals_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
        cohesion_parameters=cohesion_parameters,
    )

    generated_profile = pl.generate_profile(number_of_ballots=number_of_ballots)

    # Find ballot probs
    possible_rankings = list(it.permutations(candidates, len(candidates)))
    ballot_prob_dict = {b: 0.0 for b in possible_rankings}

    for ranking in possible_rankings:
        # ranking = b.ranking
        for bloc in bloc_voter_prop.keys():
            support_for_cands = pl.pref_interval_by_bloc[bloc].interval
            total_prob = 1
            prob = bloc_voter_prop[bloc]
            for cand in ranking:
                prob *= support_for_cands[cand] / total_prob
                total_prob -= support_for_cands[cand]
            ballot_prob_dict[ranking] += prob

    assert isinstance(generated_profile, PreferenceProfile)
    # Test
    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, generated_profile)


def test_SPL_distribution():
    # Set-up
    number_of_ballots = 500
    candidates = ["W1", "W2", "C1", "C2"]
    pref_intervals_by_bloc = {
        "W": {
            "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
            "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
        },
        "C": {
            "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
            "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
        },
    }
    bloc_voter_prop = {"W": 0.7, "C": 0.3}
    cohesion_parameters = {"W": {"W": 0.9, "C": 0.1}, "C": {"C": 0.8, "W": 0.2}}
    slate_to_candidates = {"W": ["W1", "W2"], "C": ["C1", "C2"]}

    # Generate ballots
    generated_profile_by_bloc, _ = slate_PlackettLuce(
        candidates=candidates,
        pref_intervals_by_bloc=pref_intervals_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
        cohesion_parameters=cohesion_parameters,
        slate_to_candidates=slate_to_candidates,
    ).generate_profile(
        number_of_ballots=number_of_ballots, by_bloc=True
    )  # type: ignore

    blocs = list(bloc_voter_prop.keys())

    # Find labeled ballot probs
    possible_rankings = list(it.permutations(candidates))
    for current_bloc in blocs:
        ballot_prob_dict = {b: 0.0 for b in possible_rankings}

        for ranking in possible_rankings:
            support_for_cands = combine_preference_intervals(
                list(pref_intervals_by_bloc[current_bloc].values()),
                [1 / len(blocs) for _ in range(len(blocs))],
            ).interval
            # pref_interval_by_bloc[current_bloc]
            total_prob = 1
            prob = 1
            for cand in ranking:
                prob *= support_for_cands[cand] / total_prob
                total_prob -= support_for_cands[cand]
            ballot_prob_dict[ranking] += prob

        candidate_to_slate = {
            c: s for s, c_list in slate_to_candidates.items() for c in c_list
        }
        # now compute unlabeled ballot probs and multiply by labeled ballot probs
        for ballot in ballot_prob_dict.keys():
            # relabel candidates by their bloc
            ballot_by_bloc = [candidate_to_slate[c] for c in ballot]
            prob = 1
            bloc_counter = {b: 0.0 for b in bloc_voter_prop.keys()}
            # compute prob of ballot type

            prob_mass = 1
            temp_cohesion = cohesion_parameters[current_bloc].copy()
            for bloc in ballot_by_bloc:
                prob *= temp_cohesion[bloc] / prob_mass
                bloc_counter[bloc] += 1

                # if no more of current bloc, renormalize
                if bloc_counter[bloc] == len(slate_to_candidates[bloc]):
                    del temp_cohesion[bloc]
                    prob_mass = sum(temp_cohesion.values())

                    # if only one bloc left, determined
                    if len(temp_cohesion) == 1:
                        break

            ballot_prob_dict[ballot] *= prob

        # Test
        assert do_ballot_probs_match_ballot_dist(
            ballot_prob_dict, generated_profile_by_bloc[current_bloc]
        )


def test_NBT_distribution():
    # Set-up
    number_of_ballots = 500

    candidates = ["W1", "W2", "C1", "C2"]
    pref_intervals_by_bloc = {
        "W": {
            "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
            "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
        },
        "C": {
            "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
        },
    }
    bloc_voter_prop = {"W": 0.7, "C": 0.3}
    cohesion_parameters = {"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.6, "W": 0.4}}
    bloc_voter_prop = {"W": 0.7, "C": 0.3}

    # Generate ballots
    config = BlocSlateConfig(
        n_voters=number_of_ballots,
        slate_to_candidates={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        bloc_proportions=bloc_voter_prop,
        preference_mapping=pref_intervals_by_bloc,
        cohesion_mapping=cohesion_parameters,
    )

    generated_profile = generate_name_bt_profile(config)

    # Find ballot probs
    possible_rankings = list(it.permutations(candidates, len(candidates)))

    final_ballot_prob_dict = {b: 0.0 for b in possible_rankings}
    pref_interval_by_bloc = {
        bloc: combine_preference_intervals(
            [pref_intervals_by_bloc[bloc][slate] for slate in ["W", "C"]],
            [cohesion_parameters[bloc][slate] for slate in ["W", "C"]],
        )
        for bloc in ["W", "C"]
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

    assert isinstance(generated_profile, PreferenceProfile)
    # Test
    assert do_ballot_probs_match_ballot_dist(final_ballot_prob_dict, generated_profile)


def test_NBT_3_bloc():
    slate_to_candidates = {"A": ["A1"], "B": ["B1"], "C": ["C1"]}

    cohesion_parameters = {
        "A": {"A": 0.7, "B": 0.2, "C": 0.1},
        "B": {"A": 0.7, "B": 0.2, "C": 0.1},
        "C": {"A": 0.7, "B": 0.2, "C": 0.1},
    }

    pref_intervals_by_bloc = {
        "A": {
            "A": PreferenceInterval({"A1": 1}),
            "B": PreferenceInterval({"B1": 1}),
            "C": PreferenceInterval({"C1": 1}),
        },
        "B": {
            "A": PreferenceInterval({"A1": 1}),
            "B": PreferenceInterval({"B1": 1}),
            "C": PreferenceInterval({"C1": 1}),
        },
        "C": {
            "A": PreferenceInterval({"A1": 1}),
            "B": PreferenceInterval({"B1": 1}),
            "C": PreferenceInterval({"C1": 1}),
        },
    }

    bloc_voter_prop = {"A": 0.9998, "B": 0.0001, "C": 0.0001}

    # Generate ballots
    config = BlocSlateConfig(
        n_voters=500,
        slate_to_candidates=slate_to_candidates,
        bloc_proportions=bloc_voter_prop,
        preference_mapping=pref_intervals_by_bloc,
        cohesion_mapping=cohesion_parameters,
        silent=True,
    )

    profile = generate_name_bt_profile(config)

    summ = 98 + 28 + 7 + 49 + 4 + 2

    ballot_prob_dict = {
        ("A1", "B1", "C1"): 98 / summ,
        ("A1", "C1", "B1"): 49 / summ,
        ("B1", "A1", "C1"): 28 / summ,
        ("B1", "C1", "A1"): 4 / summ,
        ("C1", "A1", "B1"): 7 / summ,
        ("C1", "B1", "A1"): 2 / summ,
    }

    assert isinstance(profile, PreferenceProfile)

    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, profile)

    alphas = {
        "A": {"A": 1, "B": 1, "C": 1},
        "B": {"A": 1, "B": 1, "C": 1},
        "C": {"A": 1, "B": 1, "C": 1},
    }

    config.set_dirichlet_alphas(alphas)

    profile = generate_name_bt_profile(config)
    assert isinstance(profile, PreferenceProfile)


def test_SPL_3_bloc():
    slate_to_candidates = {"A": ["A1", "A2"], "B": ["B1"], "C": ["C1"]}

    candidates = [c for c_list in slate_to_candidates.values() for c in c_list]

    cohesion_parameters = {
        "A": {"A": 0.7, "B": 0.2, "C": 0.1},
        "B": {"A": 0.7, "B": 0.2, "C": 0.1},
        "C": {"A": 0.7, "B": 0.2, "C": 0.1},
    }

    pref_intervals_by_bloc = {
        "A": {
            "A": PreferenceInterval({"A1": 1 / 2, "A2": 1 / 2}),
            "B": PreferenceInterval({"B1": 1}),
            "C": PreferenceInterval({"C1": 1}),
        },
        "B": {
            "A": PreferenceInterval({"A1": 1, "A2": 1}),
            "B": PreferenceInterval({"B1": 1}),
            "C": PreferenceInterval({"C1": 1}),
        },
        "C": {
            "A": PreferenceInterval({"A1": 1, "A2": 1}),
            "B": PreferenceInterval({"B1": 1}),
            "C": PreferenceInterval({"C1": 1}),
        },
    }

    bloc_voter_prop = {"A": 1, "B": 0, "C": 0}

    sp = slate_PlackettLuce(
        slate_to_candidates=slate_to_candidates,
        cohesion_parameters=cohesion_parameters,
        pref_intervals_by_bloc=pref_intervals_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
        candidates=candidates,
    )

    profile = sp.generate_profile(500)

    ballot_prob_dict = {
        ("A1", "A2", "B1", "C1"): 1 / 2 * 49 / 100 * 2 / 3,
        ("A1", "A2", "C1", "B1"): 1 / 2 * 49 / 100 * 1 / 3,
        ("A1", "B1", "A2", "C1"): 1 / 2 * 7 / 10 * 2 / 10 * 7 / 8,
        ("A1", "C1", "A2", "B1"): 1 / 2 * 7 / 10 * 1 / 10 * 7 / 9,
        ("A1", "B1", "C1", "A2"): 1 / 2 * 7 / 10 * 2 / 10 * 1 / 8,
        ("A1", "C1", "B1", "A2"): 1 / 2 * 7 / 10 * 1 / 10 * 2 / 9,
        ("B1", "C1", "A1", "A2"): 1 / 2 * 2 / 10 * 1 / 8,
        ("C1", "B1", "A1", "A2"): 1 / 2 * 1 / 10 * 2 / 9,
        ("B1", "A1", "C1", "A2"): 1 / 2 * 2 / 10 * 7 / 8 * 1 / 8,
        ("C1", "A1", "B1", "A2"): 1 / 2 * 1 / 10 * 7 / 9 * 2 / 9,
        ("B1", "A1", "A2", "C1"): 1 / 2 * 2 / 10 * 49 / 64,
        ("C1", "A1", "A2", "B1"): 1 / 2 * 1 / 10 * 49 / 81,
        ("A2", "A1", "B1", "C1"): 1 / 2 * 49 / 100 * 2 / 3,
        ("A2", "A1", "C1", "B1"): 1 / 2 * 49 / 100 * 1 / 3,
        ("A2", "B1", "A1", "C1"): 1 / 2 * 7 / 10 * 2 / 10 * 7 / 8,
        ("A2", "C1", "A1", "B1"): 1 / 2 * 7 / 10 * 1 / 10 * 7 / 9,
        ("A2", "B1", "C1", "A1"): 1 / 2 * 7 / 10 * 2 / 10 * 1 / 8,
        ("A2", "C1", "B1", "A1"): 1 / 2 * 7 / 10 * 1 / 10 * 2 / 9,
        ("B1", "C1", "A2", "A1"): 1 / 2 * 2 / 10 * 1 / 8,
        ("C1", "B1", "A2", "A1"): 1 / 2 * 1 / 10 * 2 / 9,
        ("B1", "A2", "C1", "A1"): 1 / 2 * 2 / 10 * 7 / 8 * 1 / 8,
        ("C1", "A2", "B1", "A1"): 1 / 2 * 1 / 10 * 7 / 9 * 2 / 9,
        ("B1", "A2", "A1", "C1"): 1 / 2 * 2 / 10 * 49 / 64,
        ("C1", "A2", "A1", "B1"): 1 / 2 * 1 / 10 * 49 / 81,
    }

    assert isinstance(profile, PreferenceProfile)
    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, profile)

    alphas = {
        "A": {"A": 1, "B": 1, "C": 1},
        "B": {"A": 1, "B": 1, "C": 1},
        "C": {"A": 1, "B": 1, "C": 1},
    }

    sp = slate_PlackettLuce.from_params(
        slate_to_candidates=slate_to_candidates,
        cohesion_parameters=cohesion_parameters,
        alphas=alphas,
        bloc_voter_prop=bloc_voter_prop,
        candidates=candidates,
    )

    assert len(sp.pref_intervals_by_bloc["A"]) == 3
    assert isinstance(sp.pref_intervals_by_bloc["A"]["A"], PreferenceInterval)

    profile = sp.generate_profile(3)
    assert isinstance(profile, PreferenceProfile)


def test_NBT_probability_calculation():
    # Set-up
    slate_to_candidate = {"W": ["W1", "W2"], "C": ["C1", "C2"]}
    pref_intervals_by_bloc = {
        "W": {
            "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
            "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
        },
        "C": {
            "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
        },
    }
    bloc_voter_prop = {"W": 0.7, "C": 0.3}
    cohesion_parameters = {"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.6, "W": 0.4}}

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
    ww_pref_interval = pref_interval_by_bloc["W"].interval
    cc_pref_interval = pref_interval_by_bloc["C"].interval

    assert bt_prob(
        permutations=[permutation], cand_support_dict=dict(cc_pref_interval)
    )[permutation] == (
        cc_pref_interval["W1"] / (cc_pref_interval["W1"] + cc_pref_interval["W2"])
    )

    permutation = ("W1", "W2", "C2")
    prob = (
        (ww_pref_interval["W1"] / (ww_pref_interval["W1"] + ww_pref_interval["W2"]))
        * (ww_pref_interval["W1"] / (ww_pref_interval["W1"] + ww_pref_interval["C2"]))
        * (ww_pref_interval["W2"] / (ww_pref_interval["W2"] + ww_pref_interval["C2"]))
    )
    assert (
        bt_prob(permutations=[permutation], cand_support_dict=dict(ww_pref_interval))[
            permutation
        ]
        == prob
    )


def compute_pl_prob(perm, interval):
    pref_interval = interval.copy()
    prob = 1
    for c in perm:
        if sum(pref_interval.values()) == 0:
            prob *= 1 / math.factorial(len(pref_interval))
        else:
            prob *= pref_interval[c] / sum(pref_interval.values())
        del pref_interval[c]
    return prob


def bloc_order_probs_slate_first(slate, ballot_frequencies):
    slate_first_count = sum(
        [freq for ballot, freq in ballot_frequencies.items() if ballot[0] == slate]
    )
    prob_ballot_given_slate_first = {
        ballot: freq / slate_first_count
        for ballot, freq in ballot_frequencies.items()
        if ballot[0] == slate
    }
    return prob_ballot_given_slate_first


def test_Cambridge_distribution():
    # BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = "src/votekit/ballot_generator/bloc_slate_generator/data"
    path = Path(DATA_DIR, "Cambridge_09to17_ballot_types.p")

    candidates = ["W1", "W2", "C1", "C2"]
    slate_to_candidate = {"W": ["W1", "W2"], "C": ["C1", "C2"]}
    pref_intervals_by_bloc = {
        "W": {
            "W": PreferenceInterval({"W1": 0.4, "W2": 0.4}),
            "C": PreferenceInterval({"C1": 0.1, "C2": 0.1}),
        },
        "C": {
            "W": PreferenceInterval({"W1": 0.1, "W2": 0.1}),
            "C": PreferenceInterval({"C1": 0.4, "C2": 0.4}),
        },
    }

    bloc_voter_prop = {"W": 0.5, "C": 0.5}
    cohesion_parameters = {"W": {"W": 1, "C": 0}, "C": {"C": 1, "W": 0}}

    cs = CambridgeSampler(
        candidates=candidates,
        slate_to_candidates=slate_to_candidate,
        pref_intervals_by_bloc=pref_intervals_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
        cohesion_parameters=cohesion_parameters,
        path=path,
    )

    candidates = ["W1", "W2", "C1", "C2"]
    slate_to_candidate = {"W": ["W1", "W2"], "C": ["C1", "C2"]}
    pref_interval_by_bloc = {
        "W": {"W1": 0.4, "W2": 0.4, "C1": 0.1, "C2": 0.1},
        "C": {"W1": 0.1, "W2": 0.1, "C1": 0.4, "C2": 0.4},
    }
    bloc_voter_prop = {"W": 0.5, "C": 0.5}
    cohesion_parameters = {"W": 1, "C": 1}

    with open(path, "rb") as pickle_file:
        ballot_frequencies = pickle.load(pickle_file)
    slates = list(slate_to_candidate.keys())

    # Let's update the running probability of the ballot based on where we are in the nesting
    ballot_prob_dict = dict()
    ballot_prob = [0.0, 0.0, 0.0, 0.0, 0.0]
    # p(white) vs p(poc)
    for slate in slates:
        opp_slate = next(iter(set(slates).difference(set(slate))))

        slate_cands = slate_to_candidate[slate]
        opp_cands = slate_to_candidate[opp_slate]

        ballot_prob[0] = bloc_voter_prop[slate]
        prob_ballot_given_slate_first = bloc_order_probs_slate_first(
            slate, ballot_frequencies
        )
        # p(crossover) vs p(non-crossover)
        for voter_bloc in slates:
            opp_voter_bloc = next(iter(set(slates).difference(set(voter_bloc))))
            if voter_bloc == slate:
                # ballot_prob[1] = 1 - bloc_crossover_rate[voter_bloc][opp_voter_bloc]
                ballot_prob[1] = cohesion_parameters[voter_bloc]

                # p(bloc ordering)
                for (
                    slate_first_ballot,
                    slate_ballot_prob,
                ) in prob_ballot_given_slate_first.items():
                    ballot_prob[2] = slate_ballot_prob

                    # Count number of each slate in the ballot
                    slate_ballot_count_dict = {}
                    for s, sc in slate_to_candidate.items():
                        count = sum([c == s for c in slate_first_ballot])
                        slate_ballot_count_dict[s] = min(count, len(sc))

                    # Make all possible perms with right number of slate candidates
                    slate_perms = list(
                        set(
                            [
                                p[: slate_ballot_count_dict[slate]]
                                for p in list(it.permutations(slate_cands))
                            ]
                        )
                    )
                    opp_perms = list(
                        set(
                            [
                                p[: slate_ballot_count_dict[opp_slate]]
                                for p in list(it.permutations(opp_cands))
                            ]
                        )
                    )

                    only_slate_interval = {
                        c: share
                        for c, share in pref_interval_by_bloc[voter_bloc].items()
                        if c in slate_cands
                    }
                    only_opp_interval = {
                        c: share
                        for c, share in pref_interval_by_bloc[voter_bloc].items()
                        if c in opp_cands
                    }
                    for sp in slate_perms:
                        ballot_prob[3] = compute_pl_prob(sp, only_slate_interval)
                        for op in opp_perms:
                            ballot_prob[4] = compute_pl_prob(op, only_opp_interval)

                            # ADD PROB MULT TO DICT
                            ordered_slate_cands = list(sp)
                            ordered_opp_cands = list(op)
                            ballot_ranking = []
                            for c in slate_first_ballot:
                                if c == slate:
                                    if ordered_slate_cands:
                                        ballot_ranking.append(
                                            ordered_slate_cands.pop(0)
                                        )
                                else:
                                    if ordered_opp_cands:
                                        ballot_ranking.append(ordered_opp_cands.pop(0))
                            prob = np.prod(ballot_prob)
                            ballot = tuple(ballot_ranking)
                            ballot_prob_dict[ballot] = (
                                ballot_prob_dict.get(ballot, 0) + prob
                            )
            else:
                # ballot_prob[1] = bloc_crossover_rate[voter_bloc][opp_voter_bloc]
                ballot_prob[1] = 1 - cohesion_parameters[voter_bloc]

                # p(bloc ordering)
                for (
                    slate_first_ballot,
                    slate_ballot_prob,
                ) in prob_ballot_given_slate_first.items():
                    ballot_prob[2] = slate_ballot_prob

                    # Count number of each slate in the ballot
                    slate_ballot_count_dict = {}
                    for s, sc in slate_to_candidate.items():
                        count = sum([c == s for c in slate_first_ballot])
                        slate_ballot_count_dict[s] = min(count, len(sc))

                    # Make all possible perms with right number of slate candidates
                    slate_perms = [
                        p[: slate_ballot_count_dict[slate]]
                        for p in list(it.permutations(slate_cands))
                    ]
                    opp_perms = [
                        p[: slate_ballot_count_dict[opp_slate]]
                        for p in list(it.permutations(opp_cands))
                    ]
                    only_slate_interval = {
                        c: share
                        for c, share in pref_interval_by_bloc[opp_voter_bloc].items()
                        if c in slate_cands
                    }
                    only_opp_interval = {
                        c: share
                        for c, share in pref_interval_by_bloc[opp_voter_bloc].items()
                        if c in opp_cands
                    }
                    for sp in slate_perms:
                        ballot_prob[3] = compute_pl_prob(sp, only_slate_interval)
                        for op in opp_perms:
                            ballot_prob[4] = compute_pl_prob(op, only_opp_interval)

                            # ADD PROB MULT TO DICT
                            ordered_slate_cands = list(sp)
                            ordered_opp_cands = list(op)
                            ballot_ranking = []
                            for c in slate_first_ballot:
                                if c == slate:
                                    if ordered_slate_cands:
                                        ballot_ranking.append(ordered_slate_cands.pop())
                                else:
                                    if ordered_opp_cands:
                                        ballot_ranking.append(ordered_opp_cands.pop())
                            prob = np.prod(ballot_prob)
                            ballot = tuple(ballot_ranking)
                            ballot_prob_dict[ballot] = (
                                ballot_prob_dict.get(ballot, 0) + prob
                            )

    # Now see if ballot prob dict is right
    test_profile = cs.generate_profile(number_of_ballots=5000)
    assert isinstance(test_profile, PreferenceProfile)
    assert do_ballot_probs_match_ballot_dist(
        ballot_prob_dict=ballot_prob_dict, generated_profile=test_profile
    )


def test_sample_ballot_types():
    slate_to_non_zero_candidates = {"A": ["A1", "A2"], "B": ["B1"]}
    cohesion_parameters_for_A_bloc = {"A": 0.8, "B": 0.2}

    sampled = sample_cohesion_ballot_types(
        slate_to_non_zero_candidates=slate_to_non_zero_candidates,
        num_ballots=100,
        cohesion_parameters_for_bloc=cohesion_parameters_for_A_bloc,
    )

    ballots = [Ballot([{str(c)} for c in b]) for b in sampled]
    pp = PreferenceProfile(ballots=ballots)

    ballot_prob_dict = {
        "AAB": cohesion_parameters_for_A_bloc["A"] ** 2,
        "ABA": cohesion_parameters_for_A_bloc["A"]
        * cohesion_parameters_for_A_bloc["B"],
        "BAA": cohesion_parameters_for_A_bloc["A"]
        * cohesion_parameters_for_A_bloc["B"],
    }
    # Test
    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, pp)

    slate_to_non_zero_candidates = {"A": ["A1"], "B": ["B1"], "C": ["C1"]}
    cohesion_parameters_for_A_bloc = {"A": 0.7, "B": 0.2, "C": 0.1}

    sampled = sample_cohesion_ballot_types(
        slate_to_non_zero_candidates=slate_to_non_zero_candidates,
        num_ballots=100,
        cohesion_parameters_for_bloc=cohesion_parameters_for_A_bloc,
    )

    ballots = [Ballot([{str(c)} for c in b]) for b in sampled]
    pp = PreferenceProfile(ballots=ballots)

    ballot_prob_dict = {
        "ABC": cohesion_parameters_for_A_bloc["A"]
        * cohesion_parameters_for_A_bloc["B"],
        "ACB": cohesion_parameters_for_A_bloc["A"]
        * cohesion_parameters_for_A_bloc["C"],
        "BAC": cohesion_parameters_for_A_bloc["B"]
        * cohesion_parameters_for_A_bloc["A"],
        "BCA": cohesion_parameters_for_A_bloc["B"]
        * cohesion_parameters_for_A_bloc["C"],
        "CAB": cohesion_parameters_for_A_bloc["C"]
        * cohesion_parameters_for_A_bloc["A"],
        "CBA": cohesion_parameters_for_A_bloc["C"]
        * cohesion_parameters_for_A_bloc["B"],
    }
    # Test
    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, pp)


def test_zero_cohesion_sample_ballot_types():
    slate_to_non_zero_candidates = {"A": ["A1", "A2"], "B": ["B1", "B2"]}
    cohesion_parameters_for_A_bloc = {"A": 1, "B": 0}

    sampled = sample_cohesion_ballot_types(
        slate_to_non_zero_candidates=slate_to_non_zero_candidates,
        num_ballots=100,
        cohesion_parameters_for_bloc=cohesion_parameters_for_A_bloc,
    )

    # each ballot has exactly one label per candidate
    expected_counts = {
        b: len(cands) for b, cands in slate_to_non_zero_candidates.items()
    }
    total_len = sum(expected_counts.values())
    assert all(len(s) == total_len for s in sampled)

    # only valid bloc labels appear
    valid = set(slate_to_non_zero_candidates)
    assert all(set(s).issubset(valid) for s in sampled)

    # counts per bloc match the number of candidates in that bloc
    assert all(Counter(s) == expected_counts for s in sampled)


def test_name_Cumulative_distribution():
    cumu = name_Cumulative(
        candidates=["A", "B"],
        pref_intervals_by_bloc={"W": {"W": PreferenceInterval({"A": 0.4, "B": 0.6})}},
        bloc_voter_prop={"W": 1},
        num_votes=2,
        cohesion_parameters={"W": {"W": 1}},
    )

    pp = cumu.generate_profile(number_of_ballots=100)

    ballot_prob_dict = {
        "AA": cumu.pref_interval_by_bloc["W"].interval["A"] ** 2,
        "AB": cumu.pref_interval_by_bloc["W"].interval["A"]
        * cumu.pref_interval_by_bloc["W"].interval["B"],
        "BA": cumu.pref_interval_by_bloc["W"].interval["A"]
        * cumu.pref_interval_by_bloc["W"].interval["B"],
        "BB": cumu.pref_interval_by_bloc["W"].interval["B"] ** 2,
    }

    assert isinstance(pp, PreferenceProfile)
    # Test
    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, pp)


def test_slate_BT_distribution():
    sbt = slate_BradleyTerry(
        slate_to_candidates={"A": ["X", "Y"], "B": ["Z"]},
        cohesion_parameters={"A": {"A": 0.8, "B": 0.2}, "B": {"A": 0.2, "B": 0.8}},
        bloc_voter_prop={"A": 1, "B": 0},
        pref_intervals_by_bloc={
            "A": {
                "A": PreferenceInterval({"X": 0.9, "Y": 0.1}),
                "B": PreferenceInterval({"Z": 1}),
            },
            "B": {
                "A": PreferenceInterval({"X": 0.9, "Y": 0.1}),
                "B": PreferenceInterval({"Z": 1}),
            },
        },
    )

    pp = sbt.generate_profile(number_of_ballots=100)

    ballot_prob_dict = {
        "XYZ": sbt.cohesion_parameters["A"]["A"] ** 2
        * sbt.pref_intervals_by_bloc["A"]["A"].interval["X"],
        "YXZ": sbt.cohesion_parameters["A"]["A"] ** 2
        * sbt.pref_intervals_by_bloc["A"]["A"].interval["Y"],
        "XZY": sbt.cohesion_parameters["A"]["A"]
        * sbt.cohesion_parameters["A"]["B"]
        * sbt.pref_intervals_by_bloc["A"]["A"].interval["X"],
        "YZX": sbt.cohesion_parameters["A"]["A"]
        * sbt.cohesion_parameters["A"]["B"]
        * sbt.pref_intervals_by_bloc["A"]["A"].interval["Y"],
        "ZXY": sbt.cohesion_parameters["A"]["B"] ** 2
        * sbt.pref_intervals_by_bloc["A"]["A"].interval["X"],
        "ZYX": sbt.cohesion_parameters["A"]["B"] ** 2
        * sbt.pref_intervals_by_bloc["A"]["A"].interval["X"],
    }

    assert isinstance(pp, PreferenceProfile)
    # Test
    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, pp)


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
