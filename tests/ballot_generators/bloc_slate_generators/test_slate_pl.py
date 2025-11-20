from votekit.pref_profile import RankProfile
from votekit.pref_interval import PreferenceInterval
from votekit.ballot_generator.bloc_slate_generator.model import BlocSlateConfig
from votekit.ballot_generator.bloc_slate_generator.slate_plackett_luce import (
    slate_pl_profile_generator,
    slate_pl_profiles_by_bloc_generator,
)
from collections import Counter
import pytest

PROB_THRESHOLD = 0.01


def compute_probability_of_ballot_by_slate(
    bloc: str, config: BlocSlateConfig, slate_ballot_type: list[str]
):
    """
    Compute the probability of a ballot by slate given the sPL model.

    Args:
        bloc (str): The bloc for which to compute the probability of the ballot.
        config (BlocSlateConfig): The configuration for the bloc-slate type model.
        slate_ballot_type (list[str]): The slate ballot type for which to compute the probability.

    Returns:
        float: The probability of the ballot by slate.

    Raises:
        ValueError: If a tie occurs in the ballot.
    """

    num_cands_seen_per_slate = {slate: 0 for slate in config.slates}
    num_cands_per_slate = {
        slate: len(cand_list) for slate, cand_list in config.slate_to_candidates.items()
    }
    prob_of_slate: dict[str, float] = {
        slate: float(config.cohesion_df.loc[bloc][slate]) for slate in config.slates
    }

    ballot_prob = 1.0
    for i, slate in enumerate(slate_ballot_type):
        ballot_prob *= prob_of_slate[slate]
        num_cands_seen_per_slate[slate] += 1

        if num_cands_seen_per_slate[slate] == num_cands_per_slate[slate]:
            del prob_of_slate[slate]
            prob_of_slate = {
                slate: prob / sum(prob_of_slate.values())
                for slate, prob in prob_of_slate.items()
            }

    return ballot_prob


def test_SPL_completion(two_bloc_two_slate_config):
    config = two_bloc_two_slate_config
    profile = slate_pl_profile_generator(config)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100_000

    profile_dict = slate_pl_profiles_by_bloc_generator(config)

    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["X"])) is RankProfile
    assert (type(profile_dict["Y"])) is RankProfile


def test_SPL_invalid_config():
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
        slate_pl_profile_generator(config)


def test_two_bloc_two_slate_spl_distribution_matches_slate_ballot_dist(
    two_bloc_two_slate_config,
):
    config = two_bloc_two_slate_config

    profiles_by_bloc = slate_pl_profiles_by_bloc_generator(config, group_ballots=True)

    for bloc in config.blocs:
        profile = profiles_by_bloc[bloc]
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
            slate_ballot_weight / profile.total_ballot_wt
            - compute_probability_of_ballot_by_slate(bloc, config, slate_ballot_type)
            < PROB_THRESHOLD
            for slate_ballot_type, slate_ballot_weight in slate_ballot_counts.items()
        )


def test_two_bloc_two_slate_spl_distribution_matches_name_ballot_dist(
    two_bloc_two_slate_config,
):
    config = two_bloc_two_slate_config

    profiles_by_bloc = slate_pl_profiles_by_bloc_generator(config, group_ballots=True)

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


def test_one_bloc_three_slate_spl_distribution_matches_slate_ballot_dist(
    one_bloc_three_slate_config,
):
    config = one_bloc_three_slate_config

    profile = slate_pl_profile_generator(config, group_ballots=True)

    cand_to_slate_dict = {
        cand: slate
        for slate, cand_list in config.slate_to_candidates.items()
        for cand in cand_list
    }
    slate_ballot_counts = {}
    bloc = "X"

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
        slate_ballot_weight / profile.total_ballot_wt
        - compute_probability_of_ballot_by_slate(bloc, config, slate_ballot_type)
        < PROB_THRESHOLD
        for slate_ballot_type, slate_ballot_weight in slate_ballot_counts.items()
    )


def test_one_bloc_three_slate_spl_distribution_matches_name_ballot_dist(
    one_bloc_three_slate_config,
):
    config = one_bloc_three_slate_config

    profile = slate_pl_profile_generator(config, group_ballots=True)

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


def test_spl_zero_support_slates():
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

    profile = slate_pl_profile_generator(config)
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
