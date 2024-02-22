import itertools as it
import math
import pytest
import scipy.stats as stats
from pathlib import Path
import pickle
import numpy as np

from votekit.ballot_generator import (
    ImpartialAnonymousCulture,
    ImpartialCulture,
    name_PlackettLuce,
    name_BradleyTerry,
    AlternatingCrossover,
    CambridgeSampler,
    OneDimSpatial,
    BallotSimplex,
    slate_PlackettLuce,
    slate_BradleyTerry
)
from votekit.pref_profile import PreferenceProfile
from votekit.pref_interval import PreferenceInterval, combine_preference_intervals

# set seed for more consistent tests
np.random.seed(8675309)


def test_IC_completion():
    ic = ImpartialCulture(candidates=["W1", "W2", "C1", "C2"], ballot_length=None)
    profile = ic.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile


def test_IAC_completion():
    iac = ImpartialAnonymousCulture(
        candidates=["W1", "W2", "C1", "C2"], ballot_length=None
    )
    profile = iac.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile


def test_NPL_completion():
    pl = name_PlackettLuce(
        candidates=["W1", "W2", "C1", "C2"],
        ballot_length=None,
        pref_intervals_by_bloc={
            "W": PreferenceInterval({"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1}),
            "C": PreferenceInterval({"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3}),
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
    )
    profile = pl.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile


def test_SPL_completion():
    sp = slate_PlackettLuce(
        candidates=["W1", "W2", "C1", "C2"],
        slate_to_candidates={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        pref_intervals_by_bloc={
            "W": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "C": {
                "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
        cohesion_parameters={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
    )
    profile = sp.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile

    result = sp.generate_profile(number_of_ballots=100, by_bloc=True)
    assert type(result) is tuple
    profile_dict, agg_prof = result
    assert type(profile_dict) is dict
    assert (type(profile_dict["W"])) is PreferenceProfile
    assert type(agg_prof) is PreferenceProfile


def test_NBT_completion():
    bt = name_BradleyTerry(
        candidates=["W1", "W2", "C1", "C2"],
        ballot_length=None,
        pref_intervals_by_bloc={
            "W": PreferenceInterval(
                interval={"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1}
            ),
            "C": PreferenceInterval(
                interval={"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3}
            ),
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
    )
    profile = bt.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile


def test_AC_completion():
    ac = AlternatingCrossover(
        candidates=["W1", "W2", "C1", "C2"],
        ballot_length=None,
        slate_to_candidates={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        pref_intervals_by_bloc={
            "W": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "C": {
                "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
        cohesion_parameters={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
    )
    profile = ac.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile


def test_1D_completion():
    ods = OneDimSpatial(candidates=["W1", "W2", "C1", "C2"], ballot_length=None)
    profile = ods.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile


def test_Cambridge_completion():
    cs = CambridgeSampler(
        candidates=["W1", "W2", "C1", "C2"],
        ballot_length=None,
        slate_to_candidates={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        pref_intervals_by_bloc={
            "W": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "C": {
                "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
        cohesion_parameters={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
    )
    profile = cs.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile


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
    n_ballots = generated_profile.num_ballots()
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
        ballot_weight = 0
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
    ballot_length = 4
    candidates = ["W1", "W2", "C1", "C2"]

    # Find ballot probs
    possible_rankings = it.permutations(candidates, ballot_length)
    ballot_prob_dict = {
        b: 1 / math.factorial(len(candidates)) for b in possible_rankings
    }

    # Generate ballots
    generated_profile = ImpartialCulture(
        ballot_length=ballot_length,
        candidates=candidates,
    ).generate_profile(number_of_ballots=number_of_ballots)

    # Test
    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, generated_profile)


def test_ballot_simplex_from_point():
    number_of_ballots = 1000
    ballot_length = 4
    candidates = ["W1", "W2", "C1", "C2"]
    pt = {"W1": 1 / 4, "W2": 1 / 4, "C1": 1 / 4, "C2": 1 / 4}

    possible_rankings = it.permutations(candidates, ballot_length)
    ballot_prob_dict = {
        b: 1 / math.factorial(len(candidates)) for b in possible_rankings
    }

    generated_profile = BallotSimplex.from_point(
        point=pt, ballot_length=ballot_length, candidates=candidates
    ).generate_profile(number_of_ballots=number_of_ballots)
    # Test
    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, generated_profile)


def test_ballot_simplex_from_alpha_zero():
    number_of_ballots = 1000
    candidates = ["W1", "W2", "C1", "C2"]

    generated_profile = BallotSimplex.from_alpha(
        alpha=0, candidates=candidates
    ).generate_profile(number_of_ballots=number_of_ballots)

    assert len(generated_profile.ballots) == 1


# def test_iac_distribution():
#     number_of_ballots = 1000
#     ballot_length = 4
#     candidates = ["W1", "W2", "C1", "C2"]

#     # Find ballot probs
#     possible_rankings = list(it.permutations(candidates, ballot_length))
#     probabilities = np.random.dirichlet([1] * len(possible_rankings))

#     ballot_prob_dict = {
#         possible_rankings[b_ind]: probabilities[b_ind]
#         for b_ind in range(len(possible_rankings))
#     }
#     generated_profile = IAC(
#         number_of_ballots=number_of_ballots,
#         ballot_length=ballot_length,
#         candidates=candidates,
#     ).generate_profile()

#     # Test
#     assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, generated_profile)


def test_NPL_distribution():
    # Set-up
    number_of_ballots = 100
    candidates = ["W1", "W2", "C1", "C2"]
    ballot_length = None
    pref_interval_by_bloc = {
        "W": PreferenceInterval({"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1}),
        "C": PreferenceInterval({"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3}),
    }
    bloc_voter_prop = {"W": 0.7, "C": 0.3}

    # Find ballot probs
    possible_rankings = list(it.permutations(candidates, ballot_length))

    ballot_prob_dict = {b: 0 for b in possible_rankings}

    for ranking in possible_rankings:
        # ranking = b.ranking
        for bloc in bloc_voter_prop.keys():
            support_for_cands = pref_interval_by_bloc[bloc].interval
            total_prob = 1
            prob = bloc_voter_prop[bloc]
            for cand in ranking:
                prob *= support_for_cands[cand] / total_prob
                total_prob -= support_for_cands[cand]
            ballot_prob_dict[ranking] += prob

    # Generate ballots
    generated_profile = name_PlackettLuce(
        ballot_length=ballot_length,
        candidates=candidates,
        pref_intervals_by_bloc=pref_interval_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
    ).generate_profile(number_of_ballots=number_of_ballots)

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
    ).generate_profile(number_of_ballots=number_of_ballots, by_bloc=True)

    blocs = list(bloc_voter_prop.keys())

    # Find labeled ballot probs
    possible_rankings = list(it.permutations(candidates))
    for current_bloc in blocs:
        ballot_prob_dict = {b: 0 for b in possible_rankings}

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
            bloc_counter = {b: 0 for b in bloc_voter_prop.keys()}
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
    ballot_length = 4
    candidates = ["W1", "W2", "C1", "C2"]
    ballot_length = None
    pref_interval_by_bloc = {
        "W": PreferenceInterval({"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1}),
        "C": PreferenceInterval({"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3}),
    }
    bloc_voter_prop = {"W": 0.7, "C": 0.3}

    # Find ballot probs
    possible_rankings = list(it.permutations(candidates, ballot_length))

    final_ballot_prob_dict = {b: 0 for b in possible_rankings}

    for bloc in bloc_voter_prop.keys():
        ballot_prob_dict = {b: 0 for b in possible_rankings}
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

    # Generate ballots
    generated_profile = name_BradleyTerry(
        ballot_length=ballot_length,
        candidates=candidates,
        pref_intervals_by_bloc=pref_interval_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
    ).generate_profile(number_of_ballots=number_of_ballots)

    # Test
    assert do_ballot_probs_match_ballot_dist(final_ballot_prob_dict, generated_profile)


def test_NBT_3_bloc():
    slate_to_candidates = {"A": ["A1"], "B": ["B1"], "C": ["C1"]}

    candidates = [c for c_list in slate_to_candidates.values() for c in c_list]

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

    bloc_voter_prop = {"A": 1, "B": 0, "C": 0}

    bt = name_BradleyTerry(
        slate_to_candidates=slate_to_candidates,
        cohesion_parameters=cohesion_parameters,
        pref_intervals_by_bloc=pref_intervals_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
        candidates=candidates,
    )

    profile = bt.generate_profile(500)

    summ = 98 + 28 + 7 + 49 + 4 + 2

    ballot_prob_dict = {
        ("A1", "B1", "C1"): 98 / summ,
        ("A1", "C1", "B1"): 49 / summ,
        ("B1", "A1", "C1"): 28 / summ,
        ("B1", "C1", "A1"): 4 / summ,
        ("C1", "A1", "B1"): 7 / summ,
        ("C1", "B1", "A1"): 2 / summ,
    }

    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, profile)

    alphas = {
        "A": {"A": 1, "B": 1, "C": 1},
        "B": {"A": 1, "B": 1, "C": 1},
        "C": {"A": 1, "B": 1, "C": 1},
    }

    bt = name_BradleyTerry.from_params(
        slate_to_candidates=slate_to_candidates,
        cohesion_parameters=cohesion_parameters,
        alphas=alphas,
        bloc_voter_prop=bloc_voter_prop,
        candidates=candidates,
    )

    assert len(bt.pref_intervals_by_bloc["A"]) == 3
    assert isinstance(bt.pref_intervals_by_bloc["A"]["A"], PreferenceInterval)

    profile = bt.generate_profile(3)
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
    ballot_length = 4
    candidates = ["W1", "W2", "C1", "C2"]
    ballot_length = None
    pref_interval_by_bloc = {
        "W": PreferenceInterval({"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1}),
        "C": PreferenceInterval({"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3}),
    }
    bloc_voter_prop = {"W": 0.7, "C": 0.3}

    model = name_BradleyTerry(
        ballot_length=ballot_length,
        candidates=candidates,
        pref_intervals_by_bloc=pref_interval_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
    )

    permutation = ("W1", "W2")

    w_pref_interval = pref_interval_by_bloc["W"].interval
    c_pref_interval = pref_interval_by_bloc["C"].interval

    assert model._calc_prob(
        permutations=[permutation], cand_support_dict=c_pref_interval
    )[permutation] == (
        c_pref_interval["W1"] / (c_pref_interval["W1"] + c_pref_interval["W2"])
    )

    permutation = ("W1", "W2", "C2")
    prob = (
        (w_pref_interval["W1"] / (w_pref_interval["W1"] + w_pref_interval["W2"]))
        * (w_pref_interval["W1"] / (w_pref_interval["W1"] + w_pref_interval["C2"]))
        * (w_pref_interval["W2"] / (w_pref_interval["W2"] + w_pref_interval["C2"]))
    )
    assert (
        model._calc_prob(permutations=[permutation], cand_support_dict=w_pref_interval)[
            permutation
        ]
        == prob
    )


def test_AC_distribution():
    def is_alternating(arr):
        return all(arr[i] != arr[i + 1] for i in range(len(arr) - 1))

    def group_elements_by_mapping(element_list, mapping):
        grouped_elements = {group: [] for group in mapping.values()}

        for element in element_list:
            group = mapping[element]
            if group is not None:
                grouped_elements[group].append(element)
        return grouped_elements

    # Set-up
    number_of_ballots = 1000
    ballot_length = 4
    candidates = ["W1", "W2", "C1", "C2"]
    ballot_length = None
    slate_to_candidate = {"W": ["W1", "W2"], "C": ["C1", "C2"]}

    cand_to_slate = {
        candidate: slate
        for slate, candidates in slate_to_candidate.items()
        for candidate in candidates
    }
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
    cohesion_parameters = {"W": 0.9, "C": 0}

    # Find ballot probs
    possible_rankings = list(it.permutations(candidates, ballot_length))
    ballot_prob_dict = {b: 0 for b in possible_rankings}

    for ranking in possible_rankings:
        slates_for_ranking = [cand_to_slate[cand] for cand in ranking]
        bloc = cand_to_slate[ranking[0]]
        starting_prob = 0

        if is_alternating(slates_for_ranking):
            bloc = cand_to_slate[ranking[1]]
            crossover_rate = 1 - cohesion_parameters[bloc]

            starting_prob = bloc_voter_prop[bloc] * crossover_rate

        # is bloc voter
        if set(slates_for_ranking[: len(slate_to_candidate[bloc])]) == {bloc}:
            starting_prob = bloc_voter_prop[bloc] * cohesion_parameters[bloc]

        ballot_prob_dict[ranking] = starting_prob
        slate_to_ranked_cands = group_elements_by_mapping(ranking, cand_to_slate)

        cand_support = combine_preference_intervals(
            list(pref_intervals_by_bloc[bloc].values()), [1 / 2, 1 / 2]
        ).interval

        prob = 1
        for ranked_cands in slate_to_ranked_cands.values():
            pref_interval = {
                k: cand_support[k] for k in cand_support if k in ranked_cands
            }
            pref_interval = {
                k: pref_interval[k] / sum(pref_interval.values()) for k in pref_interval
            }

            total_prob = 1
            for cand in ranked_cands:
                prob *= pref_interval[cand] / total_prob
                total_prob -= pref_interval[cand]

        ballot_prob_dict[ranking] *= prob

    cohesion_parameters = {
        "W": {"W": cohesion_parameters["W"], "C": 1 - cohesion_parameters["W"]},
        "C": {"C": cohesion_parameters["C"], "W": 1 - cohesion_parameters["C"]},
    }
    # Generate ballots
    generated_profile = AlternatingCrossover(
        ballot_length=ballot_length,
        candidates=candidates,
        pref_intervals_by_bloc=pref_intervals_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
        slate_to_candidates=slate_to_candidate,
        cohesion_parameters=cohesion_parameters,
    ).generate_profile(number_of_ballots=number_of_ballots)

    # Test
    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, generated_profile)


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


def test_setparams_npl():
    blocs = {"R": 0.6, "D": 0.4}
    cohesion = {"R": {"R": 0.7, "D": 0.3}, "D": {"D": 0.6, "R": 0.4}}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}

    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}

    pl = name_PlackettLuce.from_params(
        slate_to_candidates=slate_to_cands,
        bloc_voter_prop=blocs,
        cohesion_parameters=cohesion,
        alphas=alphas,
    )
    # check params were set
    assert pl.bloc_voter_prop == {"R": 0.6, "D": 0.4}
    interval = pl.pref_intervals_by_bloc
    # check if intervals add up to one
    for bloc in blocs:
        for b in blocs:
            assert math.isclose(sum(interval[bloc][b].interval.values()), 1)


def test_bt_single_bloc():
    blocs = {"R": 0.6, "D": 0.4}
    cohesion = {"R": {"R": 0.7, "D": 0.3}, "D": {"D": 0.6, "R": 0.4}}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}
    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}

    gen = name_BradleyTerry.from_params(
        slate_to_candidates=slate_to_cands,
        bloc_voter_prop=blocs,
        cohesion_parameters=cohesion,
        alphas=alphas,
    )
    interval = gen.pref_interval_by_bloc
    assert math.isclose(sum(interval["R"].interval.values()), 1)


def test_incorrect_blocs():
    blocs = {"R": 0.7, "D": 0.4}
    cohesion = {"R": 0.7, "D": 0.6}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}
    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}

    with pytest.raises(ValueError):
        name_PlackettLuce.from_params(
            slate_to_candidates=slate_to_cands,
            bloc_voter_prop=blocs,
            cohesion_parameters=cohesion,
            alphas=alphas,
        )


def test_ac_profile_from_params():
    blocs = {"R": 0.6, "D": 0.4}
    # cohesion = {"R": 0.7, "D": 0.6}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}
    cohesion_parameters = {"R": {"R": 0.5, "D": 0.5}, "D": {"D": 0.4, "R": 0.6}}
    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}
    ac = AlternatingCrossover.from_params(
        bloc_voter_prop=blocs,
        alphas=alphas,
        slate_to_candidates=slate_to_cands,
        cohesion_parameters=cohesion_parameters,
    )

    profile = ac.generate_profile(3)
    assert type(profile) is PreferenceProfile


def test_cambridge_profile_from_params():
    blocs = {"R": 0.6, "D": 0.4}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}
    cohesion_parameters = {"R": {"R": 0.5, "D": 0.5}, "D": {"D": 0.4, "R": 0.6}}
    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}
    ac = CambridgeSampler.from_params(
        bloc_voter_prop=blocs,
        alphas=alphas,
        slate_to_candidates=slate_to_cands,
        cohesion_parameters=cohesion_parameters,
    )

    profile = ac.generate_profile(3)
    assert type(profile) is PreferenceProfile


def test_npl_profile_from_params():
    blocs = {"R": 0.6, "D": 0.4}
    cohesion = {"R": {"R": 0.7, "D": 0.3}, "D": {"D": 0.6, "R": 0.4}}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}
    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}

    npl = name_PlackettLuce.from_params(
        bloc_voter_prop=blocs,
        slate_to_candidates=slate_to_cands,
        cohesion_parameters=cohesion,
        alphas=alphas,
    )

    profile = npl.generate_profile(3)
    assert type(profile) is PreferenceProfile


def test_interval_sum_from_params():
    blocs = {"R": 0.6, "D": 0.4}
    cohesion = {"R": {"R": 0.7, "D": 0.3}, "D": {"D": 0.6, "R": 0.4}}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}
    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}

    npl = name_PlackettLuce.from_params(
        bloc_voter_prop=blocs,
        slate_to_candidates=slate_to_cands,
        cohesion_parameters=cohesion,
        alphas=alphas,
    )
    for curr_b in npl.blocs:
        for b in npl.blocs:
            if not math.isclose(
                sum(npl.pref_intervals_by_bloc[curr_b][b].interval.values()), 1
            ):
                assert False
    assert True


def test_Cambridge_distribution():
    # BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = "src/votekit/data"
    path = Path(DATA_DIR, "Cambridge_09to17_ballot_types.p")

    candidates = ["W1", "W2", "C1", "C2"]
    ballot_length = None
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
        ballot_length=ballot_length,
        slate_to_candidates=slate_to_candidate,
        pref_intervals_by_bloc=pref_intervals_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
        cohesion_parameters=cohesion_parameters,
        path=path,
    )

    candidates = ["W1", "W2", "C1", "C2"]
    ballot_length = None
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
    ballot_prob = [0, 0, 0, 0, 0]
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
    assert do_ballot_probs_match_ballot_dist(
        ballot_prob_dict=ballot_prob_dict, generated_profile=test_profile
    )
