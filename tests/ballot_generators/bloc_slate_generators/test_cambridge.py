from votekit.pref_interval import PreferenceInterval
from votekit.ballot_generator.bloc_slate_generator.model import BlocSlateConfig
from votekit.ballot_generator.bloc_slate_generator.cambridge import (
    cambridge_profile_generator,
    cambridge_profiles_by_bloc_generator,
)
from votekit.pref_profile import RankProfile
import json
from pathlib import Path
import pytest
import hashlib
from collections import Counter
import re

PROB_THRESHOLD = 0.01


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


# ground truth hashes of the JSON files on 11/24/25
START_WITH_C_HASH = "b238e9c898b85970021a5de8229608302620924a529f49247ee747f3588a8a71"
START_WITH_W_HASH = "fcba9103dbea6a8e03ed159123e1f4ff61ad8ac012b6acaa0d0aa6240a6ad715"


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


def test_Cambridge_completion(two_bloc_two_slate_config_cambridge):
    config = two_bloc_two_slate_config_cambridge
    profile = cambridge_profile_generator(config)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == config.n_voters

    profile_dict = cambridge_profiles_by_bloc_generator(config)
    assert isinstance(profile_dict, dict)
    assert type(profile_dict["X"]) is RankProfile
    assert type(profile_dict["Y"]) is RankProfile


def test_Cambridge_invalid_config():
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates={
            "Z": ["A1", "A2"],
            "B": ["B1", "B2"],
        },  # invalid slate name
        bloc_proportions={"X": 0.7, "Y": 0.3},
        preference_mapping={
            "X": {
                "A": PreferenceInterval({"A1": 0.4, "A2": 0.3}),
                "B": PreferenceInterval({"B1": 0.2, "B2": 0.1}),
            },
            "Y": {
                "A": PreferenceInterval({"A1": 0.2, "A2": 0.2}),
                "B": PreferenceInterval({"B1": 0.3, "B2": 0.3}),
            },
        },
        cohesion_mapping={"X": {"A": 0.7, "B": 0.3}, "Y": {"B": 0.9, "A": 0.1}},
        silent=True,
    )
    with pytest.raises(
        KeyError,
        match=r"cohesion_df columns \(slates\) must be exactly \['Z', 'B'\] as defined in the 'slate_to_candidates' parameter. Got \['A', 'B'\]",
    ):
        cambridge_profile_generator(config)


def test_Cambridge_errors(
    two_bloc_two_slate_config, two_bloc_two_slate_config_cambridge
):
    config = BlocSlateConfig(
        n_voters=100_000,
        slate_to_candidates={"A": ["A1", "A2"], "B": ["B1", "B2"], "C": ["C1", "C2"]},
        bloc_proportions={"A": 2 / 3, "B": 1 / 3},
        preference_mapping={
            "A": {
                "A": PreferenceInterval({"A1": 0.4, "A2": 0.3}),
                "B": PreferenceInterval({"B1": 0.2, "B2": 0.1}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "B": {
                "A": PreferenceInterval({"A1": 0.2, "A2": 0.2}),
                "B": PreferenceInterval({"B1": 0.3, "B2": 0.3}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
        },
        cohesion_mapping={
            "A": {"A": 1, "B": 0, "C": 0},
            "B": {"A": 1 / 3, "B": 2 / 3, "C": 0},
        },
        silent=True,
    )
    with pytest.raises(
        ValueError,
        match="This model currently only supports two slates, but you passed 3",
    ):
        cambridge_profile_generator(config)

    config = BlocSlateConfig(
        n_voters=100_000,
        slate_to_candidates={"A": ["A1", "A2"], "B": ["B1", "B2"]},
        bloc_proportions={"A": 1 / 3, "B": 1 / 3, "C": 1 / 3},
        preference_mapping={
            "A": {
                "A": PreferenceInterval({"A1": 0.4, "A2": 0.3}),
                "B": PreferenceInterval({"B1": 0.2, "B2": 0.1}),
            },
            "B": {
                "A": PreferenceInterval({"A1": 0.2, "A2": 0.2}),
                "B": PreferenceInterval({"B1": 0.3, "B2": 0.3}),
            },
            "C": {
                "A": PreferenceInterval({"A1": 0.2, "A2": 0.2}),
                "B": PreferenceInterval({"B1": 0.3, "B2": 0.3}),
            },
        },
        cohesion_mapping={
            "A": {"A": 1, "B": 0},
            "B": {"A": 1 / 3, "B": 2 / 3},
            "C": {"A": 1 / 3, "B": 2 / 3},
        },
        silent=True,
    )

    with pytest.raises(
        ValueError,
        match="This model currently only supports two blocs, but you passed 3",
    ):
        cambridge_profile_generator(config)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The slates (['A', 'B']) must correspond to the blocs (['X', 'Y'])."
        ),
    ):
        cambridge_profile_generator(two_bloc_two_slate_config)

    with pytest.raises(
        ValueError,
        match="The bloc proportions are equal. You must set a majority_bloc and minority_bloc.",
    ):
        two_bloc_two_slate_config_cambridge.bloc_proportions = {"X": 0.5, "Y": 0.5}
        cambridge_profile_generator(two_bloc_two_slate_config_cambridge)

    with pytest.raises(
        ValueError,
        match=re.escape("Majority group X and minority group X must be distinct."),
    ):
        cambridge_profile_generator(
            two_bloc_two_slate_config_cambridge, majority_bloc="X", minority_bloc="X"
        )
    with pytest.raises(
        ValueError,
        match=re.escape("Majority group Z not found in config.blocs."),
    ):
        cambridge_profile_generator(
            two_bloc_two_slate_config_cambridge, majority_bloc="Z"
        )
    with pytest.raises(
        ValueError,
        match=re.escape("Minority group Z not found in config.blocs."),
    ):
        cambridge_profile_generator(
            two_bloc_two_slate_config_cambridge, minority_bloc="Z"
        )


def test_Cambridge_two_bloc_two_slate_distribution_matches_slate_ballot_dist(
    two_bloc_two_slate_config_cambridge,
):
    config = two_bloc_two_slate_config_cambridge
    DATA_DIR = "src/votekit/ballot_generator/bloc_slate_generator/data"
    with open(
        Path(
            DATA_DIR,
            "Cambridge_09to17_ballot_types_start_with_W_ballots_distribution.json",
        ),
        "r",
    ) as f:
        starts_with_W_data = json.load(f)
    with open(
        Path(
            DATA_DIR,
            "Cambridge_09to17_ballot_types_start_with_C_ballots_distribution.json",
        ),
        "r",
    ) as f:
        starts_with_C_data = json.load(f)

    data_sets = {"X": starts_with_W_data, "Y": starts_with_C_data}
    distribution_for_config_by_starting_slate = {"X": {}, "Y": {}}

    num_cands_per_slate = {
        slate: len(config.slate_to_candidates[slate]) for slate in config.slates
    }
    for starting_slate, data_set in data_sets.items():
        distribution_for_config = {}
        for ballot_type, weight in data_set.items():
            trimmed_ballot_type = ""
            slate_counts = {slate: 0 for slate in config.slates}
            for slate in ballot_type:
                config_slate = "X" if slate == "W" else "Y"  # X is maj
                if slate_counts[config_slate] < num_cands_per_slate[config_slate]:
                    trimmed_ballot_type += config_slate
                    slate_counts[config_slate] += 1
            distribution_for_config[trimmed_ballot_type] = (
                distribution_for_config.get(trimmed_ballot_type, 0) + weight
            )
        distribution_for_config_by_starting_slate[starting_slate] = (
            distribution_for_config
        )
    profile_dict = cambridge_profiles_by_bloc_generator(config)

    cand_to_slate_dict = {
        cand: slate
        for slate, cand_list in config.slate_to_candidates.items()
        for cand in cand_list
    }
    for bloc in config.blocs:
        profile = profile_dict[bloc]
        slate_ballot_counts = {}

        for ballot in profile.ballots:
            if any(len(cand_set) > 1 for cand_set in ballot.ranking):
                raise ValueError(f"Tie occurred in ballot {ballot.ranking}")
            slate_ballot_type = "".join(
                [
                    cand_to_slate_dict[cand]
                    for cand_set in ballot.ranking
                    for cand in cand_set
                ]
            )
            slate_ballot_counts[slate_ballot_type] = (
                slate_ballot_counts.get(slate_ballot_type, 0) + ballot.weight
            )
        observed_ballot_dist = {
            ballot_type: prob / profile.total_ballot_wt
            for ballot_type, prob in slate_ballot_counts.items()
        }
        joint_distribution = {
            ballot_type: prob * config.cohesion_df.loc[bloc][starts_with_slate]
            for starts_with_slate, distribution in distribution_for_config_by_starting_slate.items()
            for ballot_type, prob in distribution.items()
        }

        assert all(
            abs(obs_freq - joint_distribution[ballot_type]) < PROB_THRESHOLD
            for ballot_type, obs_freq in observed_ballot_dist.items()
        )


def test_two_bloc_two_slate_cambridge_distribution_matches_name_ballot_dist(
    two_bloc_two_slate_config_cambridge,
):
    config = two_bloc_two_slate_config_cambridge

    profiles_by_bloc = cambridge_profiles_by_bloc_generator(config, group_ballots=True)

    for bloc in config.blocs:
        profile = profiles_by_bloc[bloc]

        for slate in config.slates:
            comparisons_profile = Counter(
                [
                    tuple(
                        cand
                        for cand_set in ballot.ranking
                        for cand in cand_set
                        if cand[0] == slate
                    )
                    for ballot in profile.ballots
                    for _ in range(int(ballot.weight))
                ]
            )

            obs_prob = float(
                (
                    comparisons_profile[(f"{slate}1", f"{slate}2")]
                    + comparisons_profile[(f"{slate}1",)]
                )
                / (profile.total_ballot_wt - comparisons_profile[()])
            )
            exp_prob = float(
                config.preference_df.loc[bloc][f"{slate}1"]
                / (
                    config.preference_df.loc[bloc][f"{slate}1"]
                    + config.preference_df.loc[bloc][f"{slate}2"]
                )
            )
            assert abs(obs_prob - exp_prob) < PROB_THRESHOLD


def test_cambridge_zero_support_slates():
    config = BlocSlateConfig(
        n_voters=100_000,
        slate_to_candidates={"A": ["A1", "A2"], "B": ["B1", "B2"]},
        bloc_proportions={"A": 1 / 3, "B": 2 / 3},
        preference_mapping={
            "A": {
                "A": PreferenceInterval({"A1": 0.4, "A2": 0.3}),
                "B": PreferenceInterval({"B1": 0.2, "B2": 0.1}),
            },
            "B": {
                "A": PreferenceInterval({"A1": 0.2, "A2": 0.2}),
                "B": PreferenceInterval({"B1": 0.3, "B2": 0.3}),
            },
        },
        cohesion_mapping={"A": {"A": 1, "B": 0}, "B": {"A": 1 / 3, "B": 2 / 3}},
    )

    profile_dict = cambridge_profiles_by_bloc_generator(config)
    profile = profile_dict["A"]

    assert all("A" in list(b.ranking[0])[0] for b in profile.ballots)
