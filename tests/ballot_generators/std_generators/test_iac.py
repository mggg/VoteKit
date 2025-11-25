import itertools as it
import numpy as np
from votekit.ballot_generator import iac_profile_generator
from votekit.pref_profile import RankProfile


def test_IAC_completion():
    profile = iac_profile_generator(
        candidates=["W1", "W2", "C1", "C2"], number_of_ballots=100
    )
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100


def test_iac_distribution(do_ballot_probs_match_ballot_dist_rank_profile):
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
    generated_profile = iac_profile_generator(
        number_of_ballots=number_of_ballots,
        candidates=candidates,
    )

    assert isinstance(generated_profile, RankProfile)
    # Test
    assert do_ballot_probs_match_ballot_dist_rank_profile(
        ballot_prob_dict, generated_profile
    )


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
