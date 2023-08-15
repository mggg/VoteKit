from votekit.ballot_generator import (
    IC,
    IAC,
    PlackettLuce,
    BradleyTerry,
    AlternatingCrossover,
    CambridgeSampler,
    OneDimSpatial,
)
from votekit.profile import PreferenceProfile
from pathlib import Path
import math
import numpy as np
import itertools as it
import scipy.stats as stats
import votekit.ballot_generator as bg


def test_base_generator():
    # TODO: test ballot to pool method
    pass


def test_IC_completion():
    ic = IC(
        number_of_ballots=100, candidates=["W1", "W2", "C1", "C2"], ballot_length=None
    )
    profile = ic.generate_profile()
    assert type(profile) is PreferenceProfile


def test_IAC_completion():
    iac = IAC(
        number_of_ballots=100, candidates=["W1", "W2", "C1", "C2"], ballot_length=None
    )
    profile = iac.generate_profile()
    assert type(profile) is PreferenceProfile


def test_PL_completion():
    pl = PlackettLuce(
        number_of_ballots=100,
        candidates=["W1", "W2", "C1", "C2"],
        ballot_length=None,
        pref_interval_by_bloc={
            "W": {"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1},
            "C": {"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3},
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
    )
    profile = pl.generate_profile()
    assert type(profile) is PreferenceProfile


def test_BT_completion():
    bt = BradleyTerry(
        number_of_ballots=100,
        candidates=["W1", "W2", "C1", "C2"],
        ballot_length=None,
        pref_interval_by_bloc={
            "W": {"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1},
            "C": {"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3},
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
    )
    profile = bt.generate_profile()
    assert type(profile) is PreferenceProfile


def test_AC_completion():
    ac = AlternatingCrossover(
        number_of_ballots=100,
        candidates=["W1", "W2", "C1", "C2"],
        ballot_length=None,
        slate_to_candidate={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        pref_interval_by_bloc={
            "W": {"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1},
            "C": {"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3},
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
        bloc_crossover_rate={"W": {"C": 0.3}, "C": {"W": 0.1}},
    )
    profile = ac.generate_profile()
    assert type(profile) is PreferenceProfile


def test_Cambridge_completion():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data/"
    path = Path(DATA_DIR, "Cambridge_09to17_ballot_types.p")

    cs = CambridgeSampler(
        number_of_ballots=100,
        candidates=["W1", "W2", "C1", "C2"],
        ballot_length=None,
        slate_to_candidate={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        pref_interval_by_bloc={
            "W": {"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1},
            "C": {"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3},
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
        bloc_crossover_rate={"W": {"C": 0.3}, "C": {"W": 0.1}},
        path=path,
    )
    profile = cs.generate_profile()
    assert type(profile) is PreferenceProfile


def test_1D_completion():
    ods = OneDimSpatial(
        number_of_ballots=100, candidates=["W1", "W2", "C1", "C2"], ballot_length=None
    )
    profile = ods.generate_profile()
    assert type(profile) is PreferenceProfile


def binomial_confidence_interval(probability, n_attempts, alpha=0.95):
    # Calculate the mean and standard deviation of the binomial distribution
    mean = n_attempts * probability
    std_dev = np.sqrt(n_attempts * probability * (1 - probability))

    # print('mean', mean)
    # print('std', std_dev)

    # Calculate the confidence interval

    z_score = stats.norm.ppf((1 + alpha) / 2)  # Z-score for 99% confidence level
    margin_of_error = z_score * (std_dev)
    # print('moe', margin_of_error)
    conf_interval = (mean - margin_of_error, mean + margin_of_error)

    return conf_interval


def do_ballot_probs_match_ballot_dist(
    ballot_prob_dict: dict, generated_profile: PreferenceProfile, n: int, alpha=0.95
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

    n_factorial = math.factorial(n)
    stdev = np.sqrt(n_factorial * alpha * (1 - alpha))
    return failed < (n_factorial * (1 - alpha) + 2 * stdev)


def test_ic_distribution():
    # Set-up
    number_of_ballots = 100
    ballot_length = 4
    candidates = ["W1", "W2", "C1", "C2"]

    # Find ballot probs
    possible_rankings = it.permutations(candidates, ballot_length)
    ballot_prob_dict = {
        b: 1 / np.math.factorial(len(candidates)) for b in possible_rankings
    }

    # Generate ballots
    generated_profile = IC(
        number_of_ballots=number_of_ballots,
        ballot_length=ballot_length,
        candidates=candidates,
    ).generate_profile()

    # Test
    assert do_ballot_probs_match_ballot_dist(
        ballot_prob_dict, generated_profile, len(candidates)
    )


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


def test_PL_distribution():
    # Set-up
    number_of_ballots = 100
    ballot_length = 4
    candidates = ["W1", "W2", "C1", "C2"]
    ballot_length = None
    pref_interval_by_bloc = {
        "W": {"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1},
        "C": {"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3},
    }
    bloc_voter_prop = {"W": 0.7, "C": 0.3}

    # Find ballot probs
    possible_rankings = list(it.permutations(candidates, ballot_length))

    ballot_prob_dict = {b: 0 for b in possible_rankings}

    for ranking in possible_rankings:
        # ranking = b.ranking
        for bloc in bloc_voter_prop.keys():
            support_for_cands = pref_interval_by_bloc[bloc]
            total_prob = 1
            prob = bloc_voter_prop[bloc]
            for cand in ranking:
                prob *= support_for_cands[cand] / total_prob
                total_prob -= support_for_cands[cand]
            ballot_prob_dict[ranking] += prob

    # Generate ballots
    generated_profile = PlackettLuce(
        number_of_ballots=number_of_ballots,
        ballot_length=ballot_length,
        candidates=candidates,
        pref_interval_by_bloc=pref_interval_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
    ).generate_profile()

    # Test
    assert do_ballot_probs_match_ballot_dist(
        ballot_prob_dict, generated_profile, len(candidates)
    )


def test_BT_distribution():

    # Set-up
    number_of_ballots = 100
    ballot_length = 4
    candidates = ["W1", "W2", "C1", "C2"]
    ballot_length = None
    pref_interval_by_bloc = {
        "W": {"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1},
        "C": {"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3},
    }
    bloc_voter_prop = {"W": 0.7, "C": 0.3}

    # Find ballot probs
    possible_rankings = list(it.permutations(candidates, ballot_length))

    final_ballot_prob_dict = {b: 0 for b in possible_rankings}

    for bloc in bloc_voter_prop.keys():
        ballot_prob_dict = {b: 0 for b in possible_rankings}
        for ranking in possible_rankings:
            support_for_cands = pref_interval_by_bloc[bloc]
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
    generated_profile = BradleyTerry(
        number_of_ballots=number_of_ballots,
        ballot_length=ballot_length,
        candidates=candidates,
        pref_interval_by_bloc=pref_interval_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
    ).generate_profile()

    # Test
    assert do_ballot_probs_match_ballot_dist(
        final_ballot_prob_dict, generated_profile, len(candidates)
    )


def test_BT_probability_calculation():

    # Set-up
    number_of_ballots = 100
    ballot_length = 4
    candidates = ["W1", "W2", "C1", "C2"]
    ballot_length = None
    pref_interval_by_bloc = {
        "W": {"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1},
        "C": {"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3},
    }
    bloc_voter_prop = {"W": 0.7, "C": 0.3}

    model = bg.BradleyTerry(
        number_of_ballots=number_of_ballots,
        ballot_length=ballot_length,
        candidates=candidates,
        pref_interval_by_bloc=pref_interval_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
    )

    permutation = ("W1", "W2")

    w_pref_interval = pref_interval_by_bloc["W"]
    c_pref_interval = pref_interval_by_bloc["C"]

    assert model._calc_prob(
        permutations=[permutation], cand_support_dict=pref_interval_by_bloc["C"]
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
        model._calc_prob(
            permutations=[permutation], cand_support_dict=pref_interval_by_bloc["W"]
        )[permutation]
        == prob
    )


# def test_AC_distribution():
#     # Set-up
#     number_of_ballots = 1000
#     ballot_length = 4
#     candidates = ["W1", "W2", "C1", "C2"]
#     ballot_length = None
#     slate_to_candidate = {"W": ["W1", "W2"], "C": ["C1", "C2"]}
#     # TODO: change this to be cand to slate
#     cand_to_slate = {
#         candidate: slate
#         for slate, candidates in slate_to_candidate.items()
#         for candidate in candidates
#     }
#     pref_interval_by_bloc = {
#         "W": {"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1},
#         "C": {"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3},
#     }
#     bloc_voter_prop = {"W": 0.7, "C": 0.3}
#     bloc_crossover_rate = {"W": 0.2, "C": 0.5}

#     # Find ballot probs
#     possible_rankings = it.permutations(candidates, ballot_length)
#     # possible_ballots = [
#     #     Ballot(ranking=[{cand} for cand in ranking], weight=Fraction(1))
#     #     for ranking in possible_rankings
#     # ]
#     ballot_prob_dict = {b: 0 for b in possible_rankings}

#     for ranking in possible_rankings:
#         # ranking = b.ranking

#         # is crossover
#         if cand_to_slate[ranking[0]] != cand_to_slate[ranking[1]]:
#             bloc = cand_to_slate[ranking[1]]
#             crossover_rate = bloc_crossover_rate[bloc]
#             prob = bloc_voter_prop[bloc] * crossover_rate

#             # calculate probabiltiy with crossover (group cands and then compute pl probability?)

#         else:
#             bloc = cand_to_slate[ranking[0]]
#             support_for_cands = pref_interval_by_bloc[bloc]

#             # calculate pl probability
#             total_prob = 1
#             prob = bloc_voter_prop[bloc]
#             for cand in ranking:
#                 prob *= support_for_cands[cand] / total_prob
#                 total_prob -= support_for_cands[cand]
#             ballot_prob_dict[ranking] += prob

#     # Generate ballots
#     generated_profile = bg.AlternatingCrossover(
#         number_of_ballots=number_of_ballots,
#         ballot_length=ballot_length,
#         candidates=candidates,
#         pref_interval_by_bloc=pref_interval_by_bloc,
#         bloc_voter_prop=bloc_voter_prop,
#         slate_to_candidate=slate_to_candidate,
#         bloc_crossover_rate=bloc_crossover_rate,
#     )

#     # Test
#     assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, generated_profile)

#     """
#     take ballot
#     - ballot is either bloc or crossover
#     - if ballot is bloc: calc probability for bloc
#     - if ballot is crossover: calc probability for crossover
#     """
