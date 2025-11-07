from votekit.ballot_generator import (
    name_pl_profile_generator,
    name_pl_profiles_by_bloc_generator,
    BlocSlateConfig,
)
from votekit.pref_profile import RankProfile
from votekit.pref_interval import PreferenceInterval
import itertools as it


def test_NPL_completion():
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
    profile = name_pl_profile_generator(config)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100

    profile_dict = name_pl_profiles_by_bloc_generator(config)
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is RankProfile  # type: ignore


def test_NPL_distribution(do_ballot_probs_match_ballot_dist_rank_profile):
    # Set-up
    number_of_ballots = 100
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

    config = BlocSlateConfig(
        n_voters=number_of_ballots,
        slate_to_candidates={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        bloc_proportions=bloc_voter_prop,
        preference_mapping=pref_intervals_by_bloc,
        cohesion_mapping=cohesion_parameters,
    )

    # Generate ballots
    generated_profile = name_pl_profile_generator(config)

    # Find ballot probs
    possible_rankings = list(it.permutations(candidates, len(candidates)))
    ballot_prob_dict = {b: 0.0 for b in possible_rankings}

    pref_interval_by_bloc = config.get_combined_preference_intervals_by_bloc()

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

    assert isinstance(generated_profile, RankProfile)
    # Test
    assert do_ballot_probs_match_ballot_dist_rank_profile(
        ballot_prob_dict, generated_profile
    )
