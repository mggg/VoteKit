from votekit.pref_profile import RankProfile, rank_profile_to_ranking_dict
from votekit.pref_interval import PreferenceInterval
from votekit.ballot_generator.bloc_slate_generator.model import BlocSlateConfig
from votekit.ballot_generator.bloc_slate_generator.slate_plackett_luce import (
    slate_pl_profile_generator,
    slate_pl_profiles_by_bloc_generator,
)
from collections import Counter
import pytest

PROB_THRESHOLD = 0.05


def test_SPL_completion():
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates={"A": ["A1", "A2"], "B": ["B1", "B2"]},
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
    )
    profile = slate_pl_profile_generator(config)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100

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


def test_two_bloc_two_slate_spl_distribution_matches_slate_ballot_dist():
    config = BlocSlateConfig(
        n_voters=10_000,
        slate_to_candidates={"A": ["A1", "A2"], "B": ["B1", "B2"]},
        bloc_proportions={"X": 0.6, "Y": 0.4},
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
    )

    profiles_by_bloc = slate_pl_profiles_by_bloc_generator(config, group_ballots=True)

    x_profile = profiles_by_bloc["X"]
    x_slate_ballots = {}
    for ballot in x_profile.ballots:
        slate_string = "".join(
            [cand[0] for cand_set in ballot.ranking for cand in cand_set]
        )
        x_slate_ballots[slate_string] = (
            x_slate_ballots.get(slate_string, 0) + ballot.weight
        )

    y_profile = profiles_by_bloc["Y"]
    y_slate_ballots = {}
    for ballot in y_profile.ballots:
        slate_string = "".join(
            [cand[0] for cand_set in ballot.ranking for cand in cand_set]
        )
        y_slate_ballots[slate_string] = (
            y_slate_ballots.get(slate_string, 0) + ballot.weight
        )

    x_slate_ballot_dist = {
        "AABB": config.cohesion_df.loc["X"]["A"] ** 2,
        "ABAB": config.cohesion_df.loc["X"]["A"] ** 2
        * config.cohesion_df.loc["X"]["B"],
        "ABBA": config.cohesion_df.loc["X"]["A"]
        * config.cohesion_df.loc["X"]["B"] ** 2,
        "BABA": config.cohesion_df.loc["X"]["B"] ** 2
        * config.cohesion_df.loc["X"]["A"],
        "BAAB": config.cohesion_df.loc["X"]["B"]
        * config.cohesion_df.loc["X"]["A"] ** 2,
        "BBAA": config.cohesion_df.loc["X"]["B"] ** 2,
    }

    y_slate_ballot_dist = {
        "AABB": config.cohesion_df.loc["Y"]["A"] ** 2,
        "ABAB": config.cohesion_df.loc["Y"]["A"] ** 2
        * config.cohesion_df.loc["Y"]["B"],
        "ABBA": config.cohesion_df.loc["Y"]["A"]
        * config.cohesion_df.loc["Y"]["B"] ** 2,
        "BABA": config.cohesion_df.loc["Y"]["B"] ** 2
        * config.cohesion_df.loc["Y"]["A"],
        "BAAB": config.cohesion_df.loc["Y"]["B"]
        * config.cohesion_df.loc["Y"]["A"] ** 2,
        "BBAA": config.cohesion_df.loc["Y"]["B"] ** 2,
    }

    assert all(
        abs(weight / x_profile.total_ballot_wt - x_slate_ballot_dist[ballot])
        < PROB_THRESHOLD
        for ballot, weight in x_slate_ballots.items()
    )

    assert all(
        abs(weight / y_profile.total_ballot_wt - y_slate_ballot_dist[ballot])
        < PROB_THRESHOLD
        for ballot, weight in y_slate_ballots.items()
    )


def test_two_bloc_two_slate_spl_distribution_matches_name_ballot_dist():
    config = BlocSlateConfig(
        n_voters=10_000,
        slate_to_candidates={"A": ["A1", "A2"], "B": ["B1", "B2"]},
        bloc_proportions={"X": 0.6, "Y": 0.4},
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
    )

    profiles_by_bloc = slate_pl_profiles_by_bloc_generator(config, group_ballots=True)

    x_profile = profiles_by_bloc["X"]

    a_comparisons_x_profile = [
        tuple(
            cand for cand_set in ballot.ranking for cand in cand_set if cand[0] == "A"
        )
        for ballot in x_profile.ballots
        for _ in range(int(ballot.weight))
    ]

    b_comparisons_x_profile = [
        tuple(
            cand for cand_set in ballot.ranking for cand in cand_set if cand[0] == "B"
        )
        for ballot in x_profile.ballots
        for _ in range(int(ballot.weight))
    ]

    assert (
        abs(
            a_comparisons_x_profile.count(("A1", "A2")) / x_profile.total_ballot_wt
            - config.preference_df.loc["X"]["A1"]
            / (
                config.preference_df.loc["X"]["A1"]
                + config.preference_df.loc["X"]["A2"]
            )
        )
        < PROB_THRESHOLD
    )

    assert (
        abs(
            b_comparisons_x_profile.count(("B1", "B2")) / x_profile.total_ballot_wt
            - config.preference_df.loc["X"]["B1"]
            / (
                config.preference_df.loc["X"]["B1"]
                + config.preference_df.loc["X"]["B2"]
            )
        )
        < PROB_THRESHOLD
    )


def test_spl_zero_support_slates():
    config = BlocSlateConfig(
        n_voters=10_000,
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


def test_spl_zero_support_candidates():
    config = BlocSlateConfig(
        n_voters=10_000,
        slate_to_candidates={
            "A": ["A1", "A2", "A3", "A4"],
        },
        bloc_proportions={"X": 1},
        preference_mapping={
            "X": {
                "A": PreferenceInterval({"A1": 0.4, "A2": 0.3, "A3": 0.0, "A4": 0.0}),
            },
        },
        cohesion_mapping={"X": {"A": 1}},
        silent=True,
    )

    profile = slate_pl_profile_generator(config)
    zero_support_candidates_perms = [
        tuple(
            cand
            for cand_set in ballot.ranking
            for cand in cand_set
            if cand[1] not in ["1", "2"]
        )
        for ballot in profile.ballots
        for _ in range(int(ballot.weight))
    ]
    zero_support_candidates_perms_dist = Counter(zero_support_candidates_perms)
    zero_support_candidates_perms_dist = {
        b: w / sum(zero_support_candidates_perms_dist.values())
        for b, w in zero_support_candidates_perms_dist.items()
    }

    assert all(
        abs(prob - 1 / 2) < PROB_THRESHOLD
        for prob in zero_support_candidates_perms_dist.values()
    )


def test_one_bloc_three_slate_spl_distribution_matches_ballot_dist():
    slate_to_candidates = {"A": ["A1", "A2"], "B": ["B1"], "C": ["C1"]}

    cohesion_parameters = {
        "X": {"A": 0.7, "B": 0.2, "C": 0.1},
    }

    pref_intervals_by_bloc = {
        "X": {
            "A": PreferenceInterval({"A1": 1 / 2, "A2": 1 / 2}),
            "B": PreferenceInterval({"B1": 1}),
            "C": PreferenceInterval({"C1": 1}),
        },
    }

    bloc_voter_prop = {
        "X": 1,
    }

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

    ranking_dict = rank_profile_to_ranking_dict(profile)
    ranking_dict_dist = {
        tuple(list(cand_set)[0] for cand_set in r): w / sum(ranking_dict.values())
        for r, w in ranking_dict.items()
    }

    assert all(
        abs(prob - ballot_prob_dict[ranking]) < PROB_THRESHOLD
        for ranking, prob in ranking_dict_dist.items()
    )
