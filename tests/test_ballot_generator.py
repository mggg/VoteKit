import itertools as it
import math

import numpy as np
import pytest
import scipy.stats as stats

from votekit.ballot_generator import (
    ImpartialAnonymousCulture,
    ImpartialCulture,
    PlackettLuce,
    BradleyTerry,
    AlternatingCrossover,
    CambridgeSampler,
    OneDimSpatial,
    BallotSimplex,
)
from votekit.pref_profile import PreferenceProfile


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


def test_PL_completion():
    pl = PlackettLuce(
        candidates=["W1", "W2", "C1", "C2"],
        ballot_length=None,
        pref_interval_by_bloc={
            "W": {"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1},
            "C": {"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3},
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
    )
    profile = pl.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile


def test_BT_completion():
    bt = BradleyTerry(
        candidates=["W1", "W2", "C1", "C2"],
        ballot_length=None,
        pref_interval_by_bloc={
            "W": {"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1},
            "C": {"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3},
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
    )
    profile = bt.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile


def test_AC_completion():
    ac = AlternatingCrossover(
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
        slate_to_candidate={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        pref_interval_by_bloc={
            "W": {"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1},
            "C": {"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3},
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
        bloc_crossover_rate={"W": {"C": 0.3}, "C": {"W": 0.1}},
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

    # allow for small margin of error given confidence intereval
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
        b: 1 / math.factorial(len(candidates)) for b in possible_rankings
    }

    # Generate ballots
    generated_profile = ImpartialCulture(
        ballot_length=ballot_length,
        candidates=candidates,
    ).generate_profile(number_of_ballots=number_of_ballots)

    # Test
    assert do_ballot_probs_match_ballot_dist(
        ballot_prob_dict, generated_profile, len(candidates)
    )


def test_ballot_simplex_from_point():
    number_of_ballots = 1000
    ballot_length = 4
    candidates = ["W1", "W2", "C1", "C2"]
    pt = {"W1": 1 / 4, "W2": 1 / 4, "C1": 1 / 4, "C2": 1 / 4}

    possible_rankings = it.permutations(candidates, ballot_length)
    ballot_prob_dict = {
        b: 1 / math.factorial(len(candidates)) for b in possible_rankings
    }

    generated_profile = (
        BallotSimplex()
        .fromPoint(point=pt, ballot_length=ballot_length, candidates=candidates)
        .generate_profile(number_of_ballots=number_of_ballots)
    )
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
        ballot_length=ballot_length,
        candidates=candidates,
        pref_interval_by_bloc=pref_interval_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
    ).generate_profile(number_of_ballots=number_of_ballots)

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
        ballot_length=ballot_length,
        candidates=candidates,
        pref_interval_by_bloc=pref_interval_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
    ).generate_profile(number_of_ballots=number_of_ballots)

    # Test
    assert do_ballot_probs_match_ballot_dist(
        final_ballot_prob_dict, generated_profile, len(candidates)
    )


def test_BT_probability_calculation():

    # Set-up
    ballot_length = 4
    candidates = ["W1", "W2", "C1", "C2"]
    ballot_length = None
    pref_interval_by_bloc = {
        "W": {"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1},
        "C": {"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3},
    }
    bloc_voter_prop = {"W": 0.7, "C": 0.3}

    model = BradleyTerry(
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
    # TODO: change this to be cand to slate
    cand_to_slate = {
        candidate: slate
        for slate, candidates in slate_to_candidate.items()
        for candidate in candidates
    }
    pref_interval_by_bloc = {
        "W": {"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1},
        "C": {"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3},
    }
    bloc_voter_prop = {"W": 0.7, "C": 0.3}
    bloc_crossover_rate = {"W": {"C": 1}, "C": {"W": 1}}

    # Find ballot probs
    possible_rankings = list(it.permutations(candidates, ballot_length))
    ballot_prob_dict = {b: 0 for b in possible_rankings}

    for ranking in possible_rankings:

        slates_for_ranking = [cand_to_slate[cand] for cand in ranking]
        bloc = cand_to_slate[ranking[0]]
        starting_prob = 0

        if is_alternating(slates_for_ranking):
            bloc = cand_to_slate[ranking[1]]
            opposing_bloc = cand_to_slate[ranking[0]]
            crossover_rate = bloc_crossover_rate[bloc][opposing_bloc]

            starting_prob = bloc_voter_prop[bloc] * crossover_rate
            print("alt", starting_prob)

        # is bloc voter
        if set(slates_for_ranking[: len(slate_to_candidate[bloc])]) == {bloc}:
            starting_prob = bloc_voter_prop[bloc] * round(
                (1 - sum(bloc_crossover_rate[bloc].values()))
            )
            print("bloc", starting_prob)

        ballot_prob_dict[ranking] = starting_prob
        slate_to_ranked_cands = group_elements_by_mapping(ranking, cand_to_slate)
        cand_support = pref_interval_by_bloc[bloc]

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

    # Generate ballots
    generated_profile = AlternatingCrossover(
        ballot_length=ballot_length,
        candidates=candidates,
        pref_interval_by_bloc=pref_interval_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
        slate_to_candidate=slate_to_candidate,
        bloc_crossover_rate=bloc_crossover_rate,
    ).generate_profile(number_of_ballots)

    # Test
    assert do_ballot_probs_match_ballot_dist(
        ballot_prob_dict, generated_profile, len(candidates)
    )


# def test_Cambridge_distribution():
#     BASE_DIR = Path(__file__).resolve().parent
#     DATA_DIR = BASE_DIR / "data/"
#     path = Path(DATA_DIR, "Cambridge_09to17_ballot_types.p")
#
#     candidates = ["W1", "W2", "C1", "C2"]
#     ballot_length = None
#     slate_to_candidate = {"W": ["W1", "W2"], "C": ["C1", "C2"]}
#     pref_interval_by_bloc = {
#         "W": {"W1": 0.4, "W2": 0.4, "C1": 0.1, "C2": 0.1},
#         "C": {"W1": 0.1, "W2": 0.1, "C1": 0.4, "C2": 0.4},
#     }
#     bloc_voter_prop = {"W": 0.5, "C": 0.5}
#     bloc_crossover_rate = {"W": {"C": 0}, "C": {"W": 0}}
#
#     cs = CambridgeSampler(
#         candidates=candidates,
#         ballot_length=ballot_length,
#         slate_to_candidate=slate_to_candidate,
#         pref_interval_by_bloc=pref_interval_by_bloc,
#         bloc_voter_prop=bloc_voter_prop,
#         bloc_crossover_rate=bloc_crossover_rate,
#         path=path,
#     )
#
#     with open(path, "rb") as pickle_file:
#         ballot_frequencies = pickle.load(pickle_file)
#     slates = list(slate_to_candidate.keys())
#
#     # Let's update the running probability of the ballot based on where we are in the nesting
#     ballot_prob_dict = dict()
#     ballot_prob = [0, 0, 0, 0, 0]
#     # p(white) vs p(poc)
#     for slate in slates:
#         opp_slate = next(iter(set(slates).difference(set(slate))))
#
#         slate_cands = slate_to_candidate[slate]
#         opp_cands = slate_to_candidate[opp_slate]
#
#         ballot_prob[0] = bloc_voter_prop[slate]
#         prob_ballot_given_slate_first = bloc_order_probs_slate_first(
#             slate, ballot_frequencies
#         )
#         # p(crossover) vs p(non-crossover)
#         for voter_bloc in slates:
#             opp_voter_bloc = next(iter(set(slates).difference(set(voter_bloc))))
#             if voter_bloc == slate:
#                 ballot_prob[1] = 1 - bloc_crossover_rate[voter_bloc][opp_voter_bloc]
#
#                 # p(bloc ordering)
#                 for (
#                     slate_first_ballot,
#                     slate_ballot_prob,
#                 ) in prob_ballot_given_slate_first.items():
#                     ballot_prob[2] = slate_ballot_prob
#
#                     # Count number of each slate in the ballot
#                     slate_ballot_count_dict = {}
#                     for s, sc in slate_to_candidate.items():
#                         count = sum([c == s for c in slate_first_ballot])
#                         slate_ballot_count_dict[s] = min(count, len(sc))
#
#                     # Make all possible perms with right number of slate candidates
#                     slate_perms = list(
#                         set(
#                             [
#                                 p[: slate_ballot_count_dict[slate]]
#                                 for p in list(it.permutations(slate_cands))
#                             ]
#                         )
#                     )
#                     opp_perms = list(
#                         set(
#                             [
#                                 p[: slate_ballot_count_dict[opp_slate]]
#                                 for p in list(it.permutations(opp_cands))
#                             ]
#                         )
#                     )
#
#                     only_slate_interval = {
#                         c: share
#                         for c, share in pref_interval_by_bloc[voter_bloc].items()
#                         if c in slate_cands
#                     }
#                     only_opp_interval = {
#                         c: share
#                         for c, share in pref_interval_by_bloc[voter_bloc].items()
#                         if c in opp_cands
#                     }
#                     for sp in slate_perms:
#                         ballot_prob[3] = compute_pl_prob(sp, only_slate_interval)
#                         for op in opp_perms:
#                             ballot_prob[4] = compute_pl_prob(op, only_opp_interval)
#
#                             # ADD PROB MULT TO DICT
#                             ordered_slate_cands = list(sp)
#                             ordered_opp_cands = list(op)
#                             ballot_ranking = []
#                             for c in slate_first_ballot:
#                                 if c == slate:
#                                     if ordered_slate_cands:
#                                         ballot_ranking.append(
#                                             ordered_slate_cands.pop(0)
#                                         )
#                                 else:
#                                     if ordered_opp_cands:
#                                         ballot_ranking.append(ordered_opp_cands.pop(0))
#                             prob = np.prod(ballot_prob)
#                             ballot = tuple(ballot_ranking)
#                             ballot_prob_dict[ballot] = (
#                                 ballot_prob_dict.get(ballot, 0) + prob
#                             )
#             else:
#                 ballot_prob[1] = bloc_crossover_rate[voter_bloc][opp_voter_bloc]
#
#                 # p(bloc ordering)
#                 for (
#                     slate_first_ballot,
#                     slate_ballot_prob,
#                 ) in prob_ballot_given_slate_first.items():
#                     ballot_prob[2] = slate_ballot_prob
#
#                     # Count number of each slate in the ballot
#                     slate_ballot_count_dict = {}
#                     for s, sc in slate_to_candidate.items():
#                         count = sum([c == s for c in slate_first_ballot])
#                         slate_ballot_count_dict[s] = min(count, len(sc))
#
#                     # Make all possible perms with right number of slate candidates
#                     slate_perms = [
#                         p[: slate_ballot_count_dict[slate]]
#                         for p in list(it.permutations(slate_cands))
#                     ]
#                     opp_perms = [
#                         p[: slate_ballot_count_dict[opp_slate]]
#                         for p in list(it.permutations(opp_cands))
#                     ]
#                     only_slate_interval = {
#                         c: share
#                         for c, share in pref_interval_by_bloc[opp_voter_bloc].items()
#                         if c in slate_cands
#                     }
#                     only_opp_interval = {
#                         c: share
#                         for c, share in pref_interval_by_bloc[opp_voter_bloc].items()
#                         if c in opp_cands
#                     }
#                     for sp in slate_perms:
#                         ballot_prob[3] = compute_pl_prob(sp, only_slate_interval)
#                         for op in opp_perms:
#                             ballot_prob[4] = compute_pl_prob(op, only_opp_interval)
#
#                             # ADD PROB MULT TO DICT
#                             ordered_slate_cands = list(sp)
#                             ordered_opp_cands = list(op)
#                             ballot_ranking = []
#                             for c in slate_first_ballot:
#                                 if c == slate:
#                                     if ordered_slate_cands:
#                                         ballot_ranking.append(ordered_slate_cands.pop())
#                                 else:
#                                     if ordered_opp_cands:
#                                         ballot_ranking.append(ordered_opp_cands.pop())
#                             prob = np.prod(ballot_prob)
#                             ballot = tuple(ballot_ranking)
#                             ballot_prob_dict[ballot] = (
#                                 ballot_prob_dict.get(ballot, 0) + prob
#                             )
#
#     # Now see if ballot prob dict is right
#     test_profile = cs.generate_profile(number_of_ballots=5000)
#     assert do_ballot_probs_match_ballot_dist(
#         ballot_prob_dict=ballot_prob_dict,
#         generated_profile=test_profile,
#         n=len(candidates),
#     )


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


twobloc = {
    "blocs": {"R": 0.6, "D": 0.4},
    "cohesion": {"R": 0.7, "D": 0.6},
    "alphas": {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}},
}

cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}
cands_lst = ["A", "B", "C"]

test_slate = {"R": {"A1": 0.1, "B1": 0.5, "C1": 0.4}, "D": {"A2": 0.2, "B2": 0.5}}
test_voter_prop = {"R": 0.5, "D": 0.5}


# def test_setparams_pl():
#     pl = PlackettLuce(candidates=cands, hyperparams=twobloc)
#     # check params were set
#     assert pl.bloc_voter_prop == {"R": 0.6, "D": 0.4}
#     interval = pl.pref_interval_by_bloc
#     # check if intervals add up to one
#     assert math.isclose(sum(interval["R"].values()), 1)
#     assert math.isclose(sum(interval["D"].values()), 1)


def test_bad_cands_input():
    # construct hyperparmeters with candidates not assigned to a bloc/slate
    with pytest.raises(TypeError):
        PlackettLuce(candidates=cands_lst, hyperparams=twobloc)


def test_pl_both_inputs():
    gen = PlackettLuce(
        candidates=cands,
        pref_interval_by_bloc=test_slate,
        bloc_voter_prop=test_voter_prop,
        hyperparams=twobloc,
    )
    # check that this attribute matches hyperparam input
    assert gen.bloc_voter_prop == {"R": 0.6, "D": 0.4}


# def test_bt_single_bloc():
#     bloc = {
#         "blocs": {"R": 1.0},
#         "cohesion": {"R": 0.7},
#         "alphas": {"R": {"R": 0.5, "D": 1}},
#     }
#     cands = {"R": ["X", "Y", "Z"], "D": ["A", "B"]}

#     gen = BradleyTerry(candidates=cands, hyperparams=bloc)
#     interval = gen.pref_interval_by_bloc
#     assert math.isclose(sum(interval["R"].values()), 1)


def test_incorrect_blocs():
    params = {
        "blocs": {"R": 0.7, "D": 0.4},
        "cohesion": {"R": 0.7, "D": 0.6},
        "alphas": {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}},
    }
    cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}

    with pytest.raises(ValueError):
        PlackettLuce(candidates=cands, hyperparams=params)


def test_ac_profile_from_params():
    params = {
        "blocs": {"R": 0.6, "D": 0.4},
        "cohesion": {"R": 0.7, "D": 0.6},
        "alphas": {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}},
        "crossover": {"R": {"D": 0.5}, "D": {"R": 0.6}},
    }
    cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}
    ac = AlternatingCrossover(candidates=cands, hyperparams=params)
    ballots = ac.generate_profile(3)
    assert isinstance(ballots, PreferenceProfile)
