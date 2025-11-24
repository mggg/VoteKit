from votekit.ballot_generator import (
    slate_bt_profile_generator,
    slate_bt_profiles_by_bloc_generator,
    BlocSlateConfig,
)
from votekit.pref_profile import RankProfile
from votekit.pref_interval import PreferenceInterval
import itertools as it
import pytest
from collections import Counter
import math
import re

PROB_THRESHOLD = 0.01


def compute_sbt_slate_ballot_distribution(config: BlocSlateConfig, bloc: str):
    """
    Compute the probability distribution for ballot types for a given voter bloc.

    Args:
        config (BlocSlateConfig): The configuration for the bloc-slate type model.
        bloc (str): The voter bloc for which to compute the ballot type distribution.

    Returns:
        dict[tuple[str, ...], float]: A dictionary mapping ballot types (as tuples of
            slate names) to their probabilities.
    """
    slates_with_multiplicity = [
        slate
        for slate in config.slates
        for _ in range(len(config.slate_to_candidates[slate]))
    ]
    slate_ballot_dist = {
        slate_ballot_type: 1.0
        for slate_ballot_type in it.permutations(slates_with_multiplicity)
    }
    for slate_ballot_type in slate_ballot_dist.keys():
        for idx, slate in enumerate(slate_ballot_type):
            for other_slate in slate_ballot_type[idx + 1 :]:
                if slate != other_slate:
                    slate_ballot_dist[slate_ballot_type] *= float(
                        config.cohesion_df.loc[bloc][slate]
                        / (
                            config.cohesion_df.loc[bloc][slate]
                            + config.cohesion_df.loc[bloc][other_slate]
                        )
                    )

    return {
        slate_ballot_type: mass / sum(slate_ballot_dist.values())
        for slate_ballot_type, mass in slate_ballot_dist.items()
    }


def test_SBT_completion(two_bloc_two_slate_config):
    config = two_bloc_two_slate_config
    profile = slate_bt_profile_generator(config)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100_000

    profile_dict = slate_bt_profiles_by_bloc_generator(config)
    assert isinstance(profile_dict, dict)
    assert type(profile_dict["X"]) is RankProfile
    assert type(profile_dict["Y"]) is RankProfile


def test_SBT_invalid_config():
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
        slate_bt_profile_generator(config)


def test_SBT_memory_error():
    n = 8
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates={
            "A": [f"A{i}" for i in range(n)],
            "B": [f"B{i}" for i in range(n)],
            "C": [f"C{i}" for i in range(n)],
        },
        bloc_proportions={"X": 1},
        preference_mapping={
            "X": {
                "A": PreferenceInterval({f"A{i}": 1 for i in range(n)}),
                "B": PreferenceInterval({f"B{i}": 1 for i in range(n)}),
                "C": PreferenceInterval({f"C{i}": 1 for i in range(n)}),
            },
        },
        cohesion_mapping={"X": {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}},
    )

    n_cands = len(config.candidates)
    slate_counts = {
        slate: len(cands) for slate, cands in config.slate_to_candidates.items()
    }
    total_arrangements = math.factorial(n_cands) / math.prod(
        math.factorial(count) for count in slate_counts.values()
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Given the number of candidates and slates you have entered, there appears to be "
            f"{total_arrangements:.2e} possible ballot types. This is beyond the standard limit "
            f"of 12! = {math.factorial(12)}. Please reduce the number of candidates or use the "
            "MCMC version of this generator instead."
        ),
    ):
        slate_bt_profile_generator(config)


def test_SBT_two_bloc_two_slate_distribution_matches_slate_ballot_dist(
    two_bloc_two_slate_config,
):
    config = two_bloc_two_slate_config
    profile_dict = slate_bt_profiles_by_bloc_generator(config)

    for bloc in config.blocs:
        profile = profile_dict[bloc]
        slate_ballot_dist = compute_sbt_slate_ballot_distribution(config, bloc)
        cand_to_slate_dict = {
            cand: slate
            for slate, cand_list in config.slate_to_candidates.items()
            for cand in cand_list
        }
        slate_ballot_counts = {}

        for ballot in profile.ballots:
            if any(len(cand_set) > 1 for cand_set in ballot.ranking):
                raise ValueError(f"Tie occurred in ballot {ballot.ranking}")
            slate_ballot_type = tuple(
                [
                    cand_to_slate_dict[cand]
                    for cand_set in ballot.ranking
                    for cand in cand_set
                ]
            )
            slate_ballot_counts[slate_ballot_type] = (
                slate_ballot_counts.get(slate_ballot_type, 0) + ballot.weight
            )

        assert all(
            abs(
                slate_ballot_weight / profile.total_ballot_wt
                - slate_ballot_dist[slate_ballot_type]
            )
            < PROB_THRESHOLD
            for slate_ballot_type, slate_ballot_weight in slate_ballot_counts.items()
        )


def test_two_bloc_two_slate_sbt_distribution_matches_name_ballot_dist(
    two_bloc_two_slate_config,
):
    config = two_bloc_two_slate_config

    profiles_by_bloc = slate_bt_profiles_by_bloc_generator(config, group_ballots=True)

    for bloc in config.blocs:
        profile = profiles_by_bloc[bloc]

        a_comparisons_profile = [
            tuple(
                cand
                for cand_set in ballot.ranking
                for cand in cand_set
                if cand[0] == "A"
            )
            for ballot in profile.ballots
            for _ in range(int(ballot.weight))
        ]

        b_comparisons_profile = [
            tuple(
                cand
                for cand_set in ballot.ranking
                for cand in cand_set
                if cand[0] == "B"
            )
            for ballot in profile.ballots
            for _ in range(int(ballot.weight))
        ]

        assert (
            abs(
                a_comparisons_profile.count(("A1", "A2")) / profile.total_ballot_wt
                - config.preference_df.loc[bloc]["A1"]
                / (
                    config.preference_df.loc[bloc]["A1"]
                    + config.preference_df.loc[bloc]["A2"]
                )
            )
            < PROB_THRESHOLD
        )

        assert (
            abs(
                b_comparisons_profile.count(("B1", "B2")) / profile.total_ballot_wt
                - config.preference_df.loc[bloc]["B1"]
                / (
                    config.preference_df.loc[bloc]["B1"]
                    + config.preference_df.loc[bloc]["B2"]
                )
            )
            < PROB_THRESHOLD
        )


def test_SBT_one_bloc_three_slate_distribution_matches_slate_ballot_dist(
    one_bloc_three_slate_config,
):
    config = one_bloc_three_slate_config
    profile = slate_bt_profile_generator(config)

    bloc = config.blocs[0]
    slate_ballot_dist = compute_sbt_slate_ballot_distribution(config, bloc)
    cand_to_slate_dict = {
        cand: slate
        for slate, cand_list in config.slate_to_candidates.items()
        for cand in cand_list
    }
    slate_ballot_counts = {}

    for ballot in profile.ballots:
        if any(len(cand_set) > 1 for cand_set in ballot.ranking):
            raise ValueError(f"Tie occurred in ballot {ballot.ranking}")
        slate_ballot_type = tuple(
            [
                cand_to_slate_dict[cand]
                for cand_set in ballot.ranking
                for cand in cand_set
            ]
        )
        slate_ballot_counts[slate_ballot_type] = (
            slate_ballot_counts.get(slate_ballot_type, 0) + ballot.weight
        )

    assert all(
        abs(
            slate_ballot_weight / profile.total_ballot_wt
            - slate_ballot_dist[slate_ballot_type]
        )
        < PROB_THRESHOLD
        for slate_ballot_type, slate_ballot_weight in slate_ballot_counts.items()
    )


def test_one_bloc_three_slate_sbt_distribution_matches_name_ballot_dist(
    one_bloc_three_slate_config,
):
    config = one_bloc_three_slate_config

    profile = slate_bt_profile_generator(config, group_ballots=True)

    for slate in config.slates:
        cand_comparisons_profile = [
            tuple(
                cand
                for cand_set in ballot.ranking
                for cand in cand_set
                if cand[0] == slate and cand[-1] in ["1", "2"]
            )
            for ballot in profile.ballots
            for _ in range(int(ballot.weight))
        ]

        assert (
            abs(
                cand_comparisons_profile.count((f"{slate}1", f"{slate}2"))
                / profile.total_ballot_wt
                - config.preference_df.loc["X"][f"{slate}1"]
                / (
                    config.preference_df.loc["X"][f"{slate}1"]
                    + config.preference_df.loc["X"][f"{slate}2"]
                )
            )
            < PROB_THRESHOLD
        )


def test_sbt_zero_support_slates():
    config = BlocSlateConfig(
        n_voters=100_000,
        slate_to_candidates={"A": ["A1", "A2"], "B": ["B1", "B2"], "C": ["C1"]},
        bloc_proportions={"X": 1},
        preference_mapping={
            "X": {
                "A": PreferenceInterval({"A1": 0.4, "A2": 0.3}),
                "B": PreferenceInterval({"B1": 0.2, "B2": 0.1}),
                "C": PreferenceInterval({"C1": 1}),
            },
        },
        cohesion_mapping={"X": {"A": 1, "B": 0, "C": 0}},
    )

    profile = slate_bt_profile_generator(config)
    zero_support_slate_perms = [
        tuple(
            cand[0]
            for cand_set in ballot.ranking
            for cand in cand_set
            if cand[0] != "A"
        )
        for ballot in profile.ballots
        for _ in range(int(ballot.weight))
    ]
    zero_support_slate_perms_dist = Counter(zero_support_slate_perms)
    zero_support_slate_perms_dist = {
        b: w / sum(zero_support_slate_perms_dist.values())
        for b, w in zero_support_slate_perms_dist.items()
    }

    assert all(
        abs(prob - 1 / 3) < PROB_THRESHOLD
        for prob in zero_support_slate_perms_dist.values()
    )
