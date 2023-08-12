from fractions import Fraction
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
import votekit.ballot_generator as bg
from votekit.ballot import Ballot


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


def binomial_confidence_interval(probability, n_attempts, alpha=0.99):
    # Calculate the mean and standard deviation of the binomial distribution
    mean = n_attempts * probability
    std_dev = np.sqrt(n_attempts * probability * (1 - probability))

    # Calculate the confidence interval

    probability = alpha + 0.5 * (1 - alpha)
    z_score = math.sqrt(2) * math.erfinv(
        2 * probability - 1
    )  # Z-score for 99% confidence level
    margin_of_error = z_score * (std_dev / np.sqrt(n_attempts))
    conf_interval = (mean - margin_of_error, mean + margin_of_error)

    return conf_interval


def do_ballot_probs_match_ballot_dist(
    ballot_prob_dict: dict, generated_profile: PreferenceProfile
):
    n_ballots = generated_profile.num_ballots()
    ballot_conf_dict = {
        b: binomial_confidence_interval(p, n_attempts=n_ballots)
        for b, p in ballot_prob_dict.items()
    }
    for b in ballot_conf_dict.keys():
        if not (ballot_conf_dict[b][0] < generated_profile[b] < ballot_conf_dict[b][1]):
            return False
    return True


def test_ic_distribution():
    # Set-up
    number_of_ballots = 1000
    ballot_length = 4
    candidates = ["A", "B", "C", "D"]

    # Find ballot probs
    possible_rankings = it.permutations(candidates, ballot_length)
    possible_ballots = [
        Ballot(ranking=[{cand} for cand in ranking], weight=Fraction(1))
        for ranking in possible_rankings
    ]
    ballot_prob_dict = {b: 1 / 6 for b in possible_ballots}

    # Generate ballots
    generated_profile = IC(
        number_of_ballots=number_of_ballots,
        ballot_length=ballot_length,
        candidates=candidates,
    )

    # Test
    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, generated_profile)


def test_iac_completion():
    number_of_ballots = 1000
    ballot_length = 4
    candidates = ["A", "B", "C", "D"]

    # Find ballot probs
    possible_rankings = list(it.permutations(candidates, ballot_length))
    probabilities = np.random.dirichlet([1] * len(possible_rankings))

    possible_ballots = [
        Ballot(ranking=[{cand} for cand in ranking], weight=Fraction(1))
        for ranking in possible_rankings
    ]
    ballot_prob_dict = {
        possible_ballots[b_ind]: probabilities[b_ind]
        for b_ind in range(len(possible_ballots))
    }
    generated_profile = IAC(
        number_of_ballots=number_of_ballots,
        ballot_length=ballot_length,
        candidates=candidates,
    )

    # Test
    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, generated_profile)


def test_PL_distribution():
    # Set-up
    number_of_ballots = 1000
    ballot_length = 4
    candidates = ["A", "B", "C", "D"]
    ballot_length = None
    pref_interval_by_bloc = {
        "W": {"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1},
        "C": {"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3},
    }
    bloc_voter_prop = {"W": 0.7, "C": 0.3}

    # Find ballot probs
    possible_rankings = it.permutations(candidates, ballot_length)
    possible_ballots = [
        Ballot(ranking=[{cand} for cand in ranking], weight=Fraction(1))
        for ranking in possible_rankings
    ]
    ballot_prob_dict = {b: 0 for b in possible_ballots}

    for b in possible_ballots:
        ranking = b.ranking
        for bloc in bloc_voter_prop.keys():
            support_for_cands = pref_interval_by_bloc[bloc]
            total_prob = 1
            prob = bloc_voter_prop[bloc]
            for cand in ranking:
                prob *= support_for_cands[cand] / total_prob
                total_prob -= support_for_cands[cand]
            ballot_prob_dict[b] += prob

    # Generate ballots
    generated_profile = PlackettLuce(
        number_of_ballots=number_of_ballots,
        ballot_length=ballot_length,
        candidates=candidates,
        pref_interval_by_bloc=pref_interval_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
    )

    # Test
    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, generated_profile)


def test_BT_distribution():

    # Set-up
    number_of_ballots = 1000
    ballot_length = 4
    candidates = ["A", "B", "C", "D"]
    ballot_length = None
    pref_interval_by_bloc = {
        "W": {"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1},
        "C": {"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3},
    }
    bloc_voter_prop = {"W": 0.7, "C": 0.3}

    # Find ballot probs
    possible_rankings = it.permutations(candidates, ballot_length)
    possible_ballots = [
        Ballot(ranking=[{cand} for cand in ranking], weight=Fraction(1))
        for ranking in possible_rankings
    ]
    ballot_prob_dict = {b: 0 for b in possible_ballots}

    for b in possible_ballots:
        ranking = b.ranking
        for bloc in bloc_voter_prop.keys():
            support_for_cands = pref_interval_by_bloc[bloc]
            prob = bloc_voter_prop[bloc]
            for i in range(len(ranking)):
                greater_cand = support_for_cands[ranking[i]]
                for j in range(i, len(ranking)):
                    cand = support_for_cands[ranking[j]]
                    prob *= greater_cand / (greater_cand + cand)
            ballot_prob_dict[b] += prob

        normalizer = 1 / sum(ballot_prob_dict.values())
        ballot_prob_dict = {k: v * normalizer for k, v in ballot_prob_dict.items()}

    # Generate ballots
    generated_profile = BradleyTerry(
        number_of_ballots=number_of_ballots,
        ballot_length=ballot_length,
        candidates=candidates,
        pref_interval_by_bloc=pref_interval_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
    )

    # Test
    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, generated_profile)


def test_BT_probability_calculation():

    # Set-up
    number_of_ballots = 1000
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

    permutation = ("W1", "W2", "C1", "C2")

    model._calc_prob(
        permutations=[permutation], cand_support_dict=pref_interval_by_bloc["C"]
    )


def test_AC_distribution():
    # Set-up
    number_of_ballots = 1000
    ballot_length = 4
    candidates = ["A", "B", "C", "D"]
    ballot_length = None
    slate_to_candidate = {"W": ["A", "B"], "C": ["C", "D"]}
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
    bloc_crossover_rate = {"W": 0.2, "C": 0.5}

    # Find ballot probs
    possible_rankings = it.permutations(candidates, ballot_length)
    possible_ballots = [
        Ballot(ranking=[{cand} for cand in ranking], weight=Fraction(1))
        for ranking in possible_rankings
    ]
    ballot_prob_dict = {b: 0 for b in possible_ballots}

    for b in possible_ballots:
        ranking = b.ranking

        # is crossover
        if cand_to_slate[ranking[0]] != cand_to_slate[ranking[1]]:
            bloc = cand_to_slate[ranking[1]]
            crossover_rate = bloc_crossover_rate[bloc]
            prob = bloc_voter_prop[bloc] * crossover_rate

            # calculate probabiltiy with crossover (group cands and then compute pl probability?)

        else:
            bloc = cand_to_slate[ranking[0]]
            support_for_cands = pref_interval_by_bloc[bloc]

            # calculate pl probability
            total_prob = 1
            prob = bloc_voter_prop[bloc]
            for cand in ranking:
                prob *= support_for_cands[cand] / total_prob
                total_prob -= support_for_cands[cand]
            ballot_prob_dict[b] += prob

    # Generate ballots
    generated_profile = bg.AlternatingCrossover(
        number_of_ballots=number_of_ballots,
        ballot_length=ballot_length,
        candidates=candidates,
        pref_interval_by_bloc=pref_interval_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
        slate_to_candidate=slate_to_candidate,
        bloc_crossover_rate=bloc_crossover_rate,
    )

    # Test
    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, generated_profile)

    """
    take ballot
    - ballot is either bloc or crossover
    - if ballot is bloc: calc probability for bloc
    - if ballot is crossover: calc probability for crossover
    """


# if __name__ == "__main__":
#     test_IC_completion()
#     test_IAC_completion()
#     test_PL_completion()
#     test_BT_completion()
#     test_AC_completion()
#     test_Cambridge_completion()
#     test_1D_completion()
