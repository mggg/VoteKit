from votekit.ballot_generator import ic_profile_generator
from votekit.pref_profile import RankProfile
import itertools as it
import math


def test_IC_completion():
    profile = ic_profile_generator(
        candidates=["W1", "W2", "C1", "C2"], number_of_ballots=100
    )
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100


def test_ic_distribution(do_ballot_probs_match_ballot_dist_rank_profile):
    # Set-up
    number_of_ballots = 100

    candidates = ["W1", "W2", "C1", "C2"]

    # Find ballot probs
    possible_rankings = it.permutations(candidates, len(candidates))
    ballot_prob_dict = {
        b: 1 / math.factorial(len(candidates)) for b in possible_rankings
    }

    # Generate ballots
    generated_profile = ic_profile_generator(
        candidates=candidates, number_of_ballots=number_of_ballots
    )

    assert isinstance(generated_profile, RankProfile)
    # Test
    assert do_ballot_probs_match_ballot_dist_rank_profile(
        ballot_prob_dict, generated_profile
    )
