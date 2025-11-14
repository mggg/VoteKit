from votekit.pref_interval import PreferenceInterval
from votekit.ballot_generator.bloc_slate_generator.model import BlocSlateConfig
from votekit.ballot_generator.bloc_slate_generator.cambridge import (
    cambridge_profile_generator,
    cambridge_profiles_by_bloc_generator,
)
from votekit.pref_profile import RankProfile
import json
import itertools as it
import numpy as np
from pathlib import Path
import pytest
import hashlib


def hash_json(file_path: str):
    """
    Hash a JSON file. Used to ensures that the keys and values of the JSON file are the same
    across versions.

    Args:
        file_path (str): The path to the JSON file to hash.

    Returns:
        str: The SHA-256 hash of the JSON file.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    normalized_json = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized_json.encode("utf-8")).hexdigest()


# ground truth hashes of the JSON files
START_WITH_C_HASH = "b238e9c898b85970021a5de8229608302620924a529f49247ee747f3588a8a71"
START_WITH_W_HASH = "fcba9103dbea6a8e03ed159123e1f4ff61ad8ac012b6acaa0d0aa6240a6ad715"


def test_Cambridge_distribution(
    do_ballot_probs_match_ballot_dist_rank_profile,
    compute_pl_prob,
    bloc_order_probs_slate_first,
):
    DATA_DIR = "src/votekit/ballot_generator/bloc_slate_generator/data"
    maj_path = Path(
        DATA_DIR, "Cambridge_09to17_ballot_types_start_with_W_ballots_distribution.json"
    )
    min_path = Path(
        DATA_DIR, "Cambridge_09to17_ballot_types_start_with_C_ballots_distribution.json"
    )

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

    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates=slate_to_candidate,
        bloc_proportions=bloc_voter_prop,
        preference_mapping=pref_intervals_by_bloc,
        cohesion_mapping=cohesion_parameters,
    )

    slate_to_candidate = {"W": ["W1", "W2"], "C": ["C1", "C2"]}
    pref_interval_by_bloc = {
        "W": {"W1": 0.4, "W2": 0.4, "C1": 0.1, "C2": 0.1},
        "C": {"W1": 0.1, "W2": 0.1, "C1": 0.4, "C2": 0.4},
    }
    bloc_voter_prop = {"W": 0.5, "C": 0.5}
    cohesion_parameters = {"W": 1, "C": 1}

    with open(maj_path, "r") as json_file:
        w_ballot_frequencies = json.load(json_file)
    with open(min_path, "r") as json_file:
        c_ballot_frequencies = json.load(json_file)

    ballot_frequencies = {**w_ballot_frequencies, **c_ballot_frequencies}
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
    test_profile = cambridge_profile_generator(
        config, majority_bloc="W", minority_bloc="C"
    )
    assert isinstance(test_profile, RankProfile)
    assert do_ballot_probs_match_ballot_dist_rank_profile(
        ballot_prob_dict=ballot_prob_dict, generated_profile=test_profile  # type: ignore
    )


def test_Cambridge_majority_minority_errors():
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates={"A": ["W1", "W2"], "B": ["C1", "C2"]},
        bloc_proportions={"A": 0.7, "B": 0.3},
        preference_mapping={
            "A": {
                "A": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "B": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "B": {
                "A": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "B": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        cohesion_mapping={
            "A": {"A": 0.7, "B": 0.3},
            "B": {"B": 0.9, "A": 0.1},
        },
    )

    with pytest.raises(ValueError, match="Majority group C not found in config.blocs."):
        cambridge_profile_generator(config, majority_bloc="C")

    with pytest.raises(
        ValueError, match="Majority group A and minority group A must be distinct."
    ):
        cambridge_profile_generator(config, majority_bloc="A", minority_bloc="A")

    with pytest.raises(ValueError, match="Minority group C not found in config.blocs."):
        cambridge_profile_generator(config, minority_bloc="C")

    config.bloc_proportions = {"A": 0.5, "B": 0.5}
    with pytest.raises(
        ValueError,
        match="The bloc proportions are equal. You must set a majority_bloc and minority_bloc.",
    ):
        cambridge_profile_generator(config)


def test_Cambridge_completion():
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates={"A": ["W1", "W2"], "B": ["C1", "C2"]},
        bloc_proportions={"A": 0.7, "B": 0.3},
        preference_mapping={
            "A": {
                "A": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "B": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "B": {
                "A": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "B": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        cohesion_mapping={"A": {"A": 0.7, "B": 0.3}, "B": {"B": 0.9, "A": 0.1}},
    )
    profile = cambridge_profile_generator(config)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100

    profile_dict = cambridge_profiles_by_bloc_generator(config)
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["A"])) is RankProfile


def test_Cambridge_completion_W_C_bloc():
    # W as majority
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates={"A": ["W1", "W2"], "B": ["C1", "C2"]},
        bloc_proportions={"A": 0.7, "B": 0.3},
        preference_mapping={
            "A": {
                "A": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "B": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "B": {
                "A": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "B": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        cohesion_mapping={"A": {"A": 0.7, "B": 0.3}, "B": {"B": 0.9, "A": 0.1}},
    )
    profile = cambridge_profile_generator(
        config,
        majority_bloc="A",
        minority_bloc="B",
    )
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100

    profile_dict = cambridge_profiles_by_bloc_generator(
        config,
        majority_bloc="A",
        minority_bloc="B",
    )
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["A"])) is RankProfile

    # W as minority
    profile = cambridge_profile_generator(
        config,
        majority_bloc="B",
        minority_bloc="A",
    )
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100

    profile_dict = cambridge_profiles_by_bloc_generator(
        config,
        majority_bloc="B",
        minority_bloc="A",
    )
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["A"])) is RankProfile


def test_Cambridge_json_files_are_the_same():
    DATA_DIR = "src/votekit/ballot_generator/bloc_slate_generator/data"
    assert (
        hash_json(
            Path(
                DATA_DIR,
                "Cambridge_09to17_ballot_types_start_with_W_ballots_distribution.json",
            )
        )
        == START_WITH_W_HASH
    )
    assert (
        hash_json(
            Path(
                DATA_DIR,
                "Cambridge_09to17_ballot_types_start_with_C_ballots_distribution.json",
            )
        )
        == START_WITH_C_HASH
    )
