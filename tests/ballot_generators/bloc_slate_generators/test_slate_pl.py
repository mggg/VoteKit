from votekit.pref_profile import RankProfile
from votekit.pref_interval import PreferenceInterval
import itertools as it
from votekit.ballot_generator.bloc_slate_generator.model import BlocSlateConfig
from votekit.ballot_generator.bloc_slate_generator.slate_plackett_luce import (
    slate_pl_profile_generator,
    slate_pl_profiles_by_bloc_generator,
)
from votekit.pref_interval import combine_preference_intervals
from collections import Counter
from votekit.ballot_generator.bloc_slate_generator.slate_plackett_luce import (
    _sample_pl_slate_ballots,
)
from votekit.ballot import RankBallot


def test_SPL_distribution(do_ballot_probs_match_ballot_dist_rank_profile):
    # Set-up
    number_of_ballots = 500
    candidates = ["W1", "W2", "C1", "C2"]
    pref_intervals_by_bloc = {
        "X": {
            "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
            "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
        },
        "Y": {
            "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
            "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
        },
    }
    bloc_voter_prop = {"X": 0.7, "Y": 0.3}
    cohesion_parameters = {"X": {"W": 0.9, "C": 0.1}, "Y": {"C": 0.8, "W": 0.2}}
    slate_to_candidates = {"W": ["W1", "W2"], "C": ["C1", "C2"]}

    config = BlocSlateConfig(
        n_voters=number_of_ballots,
        slate_to_candidates=slate_to_candidates,
        bloc_proportions=bloc_voter_prop,
        preference_mapping=pref_intervals_by_bloc,
        cohesion_mapping=cohesion_parameters,
    )

    generated_profile_by_bloc = slate_pl_profiles_by_bloc_generator(config)

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
            ballot_by_slate = [candidate_to_slate[c] for c in ballot]
            prob = 1
            slate_counter = {s: 0.0 for s in slate_to_candidates.keys()}
            # compute prob of ballot type

            prob_mass = 1
            temp_cohesion = cohesion_parameters[current_bloc].copy()
            for slate in ballot_by_slate:
                prob *= temp_cohesion[slate] / prob_mass
                slate_counter[slate] += 1

                # if no more of current bloc, renormalize
                if slate_counter[slate] == len(slate_to_candidates[slate]):
                    del temp_cohesion[slate]
                    prob_mass = sum(temp_cohesion.values())

                    # if only one bloc left, determined
                    if len(temp_cohesion) == 1:
                        break

            ballot_prob_dict[ballot] *= prob

        # Test
        assert do_ballot_probs_match_ballot_dist_rank_profile(
            ballot_prob_dict, generated_profile_by_bloc[current_bloc]
        )


def test_SPL_3_bloc(do_ballot_probs_match_ballot_dist_rank_profile):
    slate_to_candidates = {"A": ["A1", "A2"], "B": ["B1"], "C": ["C1"]}

    cohesion_parameters = {
        "X": {"A": 0.7, "B": 0.2, "C": 0.1},
        "Y": {"A": 0.7, "B": 0.2, "C": 0.1},
        "Z": {"A": 0.7, "B": 0.2, "C": 0.1},
    }

    pref_intervals_by_bloc = {
        "X": {
            "A": PreferenceInterval({"A1": 1 / 2, "A2": 1 / 2}),
            "B": PreferenceInterval({"B1": 1}),
            "C": PreferenceInterval({"C1": 1}),
        },
        "Y": {
            "A": PreferenceInterval({"A1": 1, "A2": 1}),
            "B": PreferenceInterval({"B1": 1}),
            "C": PreferenceInterval({"C1": 1}),
        },
        "Z": {
            "A": PreferenceInterval({"A1": 1, "A2": 1}),
            "B": PreferenceInterval({"B1": 1}),
            "C": PreferenceInterval({"C1": 1}),
        },
    }

    bloc_voter_prop = {"X": 0.999998, "Y": 0.000001, "Z": 0.000001}

    config = BlocSlateConfig(
        n_voters=500,
        slate_to_candidates=slate_to_candidates,
        bloc_proportions=bloc_voter_prop,
        preference_mapping=pref_intervals_by_bloc,
        cohesion_mapping=cohesion_parameters,
        silent=True,
    )

    profile = slate_pl_profile_generator(config)

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

    assert isinstance(profile, RankProfile)
    assert do_ballot_probs_match_ballot_dist_rank_profile(ballot_prob_dict, profile)

    config = BlocSlateConfig(
        n_voters=500,
        slate_to_candidates=slate_to_candidates,
        bloc_proportions=bloc_voter_prop,
        preference_mapping=pref_intervals_by_bloc,
        cohesion_mapping=cohesion_parameters,
        silent=True,
    )

    profile = slate_pl_profile_generator(config)

    assert isinstance(profile, RankProfile)


def test_sample_ballot_types(do_ballot_probs_match_ballot_dist_rank_profile):

    slate_to_candidates = {
        "A": ["A1", "A2"],
        "B": ["B1"],
    }
    preference_mapping = {
        "A": {
            "A": PreferenceInterval({"A1": 0.8, "A2": 0.2}),
            "B": PreferenceInterval({"B1": 1}),
        },
        "B": {
            "A": PreferenceInterval({"A1": 0.5, "A2": 0.5}),
            "B": PreferenceInterval({"B1": 0.5}),
        },
    }

    bloc_proportions = {"A": 0.8, "B": 0.2}
    cohesion_mapping = {
        "A": {"A": 0.8, "B": 0.2},
        "B": {"B": 0.9, "A": 0.1},
    }

    config = BlocSlateConfig(
        n_voters=100,
        preference_mapping=preference_mapping,
        bloc_proportions=bloc_proportions,
        slate_to_candidates=slate_to_candidates,
        cohesion_mapping=cohesion_mapping,
    )

    sampled = _sample_pl_slate_ballots(
        config=config,
        bloc="A",
        non_zero_slate_set={"A"},
        num_ballots=80,
    )

    ballots = [RankBallot(ranking=[{str(c)} for c in b]) for b in sampled]  # type: ignore
    pp = RankProfile(ballots=ballots)
    cohesion_parameters_for_A_bloc = config.cohesion_df.loc["A"]
    ballot_prob_dict = {
        "AAB": cohesion_parameters_for_A_bloc["A"] ** 2,
        "ABA": cohesion_parameters_for_A_bloc["A"]
        * cohesion_parameters_for_A_bloc["B"],
        "BAA": cohesion_parameters_for_A_bloc["A"]
        * cohesion_parameters_for_A_bloc["B"],
    }
    # Test
    assert do_ballot_probs_match_ballot_dist_rank_profile(ballot_prob_dict, pp)  # type: ignore

    cohesion_parameters_for_A_bloc = {"A": 0.7, "B": 0.2, "C": 0.1}

    sampled = _sample_pl_slate_ballots(
        config=config,
        bloc="A",
        non_zero_slate_set={"A", "B"},
        num_ballots=80,
    )

    ballots = [RankBallot(ranking=[{str(c)} for c in b]) for b in sampled]  # type: ignore
    pp = RankProfile(ballots=ballots)

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
    assert do_ballot_probs_match_ballot_dist_rank_profile(ballot_prob_dict, pp)


def test_zero_cohesion_sample_ballot_types():
    slate_to_candidates = {
        "A": ["A1", "A2"],
        "B": ["B1"],
    }
    preference_mapping = {
        "A": {
            "A": PreferenceInterval({"A1": 0.8, "A2": 0.2}),
            "B": PreferenceInterval({"B1": 1}),
        },
        "B": {
            "A": PreferenceInterval({"A1": 0.5, "A2": 0.5}),
            "B": PreferenceInterval({"B1": 0.5}),
        },
    }

    bloc_proportions = {"A": 0.8, "B": 0.2}
    cohesion_mapping = {
        "A": {"A": 1, "B": 0},
        "B": {"B": 0.9, "A": 0.1},
    }

    config = BlocSlateConfig(
        n_voters=100,
        preference_mapping=preference_mapping,
        bloc_proportions=bloc_proportions,
        slate_to_candidates=slate_to_candidates,
        cohesion_mapping=cohesion_mapping,
    )

    non_zero_slate_set = {"A"}
    sampled = _sample_pl_slate_ballots(
        config=config,
        bloc="A",
        non_zero_slate_set=non_zero_slate_set,
        num_ballots=80,
    )

    # each ballot has exactly one label per candidate
    expected_counts = {s: len(slate_to_candidates[s]) for s in non_zero_slate_set}
    total_len = sum(expected_counts.values())
    assert all(len(s) == total_len for s in sampled)

    # only valid bloc labels appear
    valid = set(slate_to_candidates)
    assert all(set(s).issubset(valid) for s in sampled)

    # counts per bloc match the number of candidates in that bloc
    assert all(Counter(s) == expected_counts for s in sampled)


def test_SPL_completion():
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
    profile = slate_pl_profile_generator(config)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100

    profile_dict = slate_pl_profiles_by_bloc_generator(config)

    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is RankProfile


def test_SPL_completion_zero_cand():
    """
    Ensure that SPL can handle candidates with 0 support.
    """
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        bloc_proportions={"W": 0.7, "C": 0.3},
        preference_mapping={
            "W": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0}),
            },
            "C": {
                "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        cohesion_mapping={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
    )
    profile = slate_pl_profile_generator(config)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100

    profile_dict = slate_pl_profiles_by_bloc_generator(config)
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is RankProfile
