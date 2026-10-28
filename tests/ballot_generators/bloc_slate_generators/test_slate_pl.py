from collections import Counter
from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest

from votekit.ballot import RankBallot
from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig
from votekit.ballot_generator.bloc_slate_generator.slate_plackett_luce import (
    _iter_slate_sequence_batches,
    slate_pl_profile_generator,
    slate_pl_profiles_by_bloc_generator,
)
from votekit.pref_interval import PreferenceInterval
from votekit.pref_profile import RankProfile
from votekit.types import Candidate

PROB_THRESHOLD = 0.01


@pytest.mark.parametrize("scale", [1.0, 1e-320])
def test_slate_pl_accumulates_waiting_times_within_each_slate(scale):
    class FixedDraws:
        def standard_exponential(self, size):
            assert size == (2, 3)
            return np.array([[1.0, 1.0, 1.5], [3.0, 0.5, 1.0]])

    sequences = np.concatenate(
        list(
            _iter_slate_sequence_batches(
                np.array([2, 1]),
                np.array([scale, scale]),
                n_voters=2,
                rng=cast(np.random.Generator, FixedDraws()),
            )
        )
    )

    np.testing.assert_array_equal(sequences, [[0, 1, 0], [1, 0, 0]])


@pytest.mark.parametrize("cohesion", [(0.5, 0.5), (0.6, 0.4)])
def test_slate_pl_retains_cohesion_until_exhaustion(cohesion):
    sequences = np.concatenate(
        list(
            _iter_slate_sequence_batches(
                np.array([2, 1]), np.array(cohesion), n_voters=30_001, rng=np.random.default_rng(42)
            )
        )
    )
    observed = Counter(map(tuple, sequences))
    expected = {
        (0, 0, 1): cohesion[0] ** 2,
        (0, 1, 0): cohesion[0] * cohesion[1],
        (1, 0, 0): cohesion[1],
    }

    assert set(observed) == set(expected)
    for sequence, probability in expected.items():
        assert observed[sequence] / len(sequences) == pytest.approx(probability, abs=0.01)


@pytest.mark.parametrize("n_slates", [1, 20])
@pytest.mark.parametrize("n_voters", [0, 17])
def test_slate_pl_preserves_counts_across_batches(monkeypatch, n_slates, n_voters):
    monkeypatch.setattr(
        "votekit.ballot_generator.bloc_slate_generator.slate_plackett_luce.BATCH_SIZE", 7
    )
    counts = np.arange(1, n_slates + 1)
    batches = list(
        _iter_slate_sequence_batches(
            counts, np.ones(n_slates), n_voters=n_voters, rng=np.random.default_rng(42)
        )
    )
    sequences = np.concatenate(batches) if batches else np.empty((0, int(counts.sum())), dtype=int)
    expected = np.repeat(np.arange(n_slates), counts)

    assert sequences.shape == (n_voters, int(counts.sum()))
    np.testing.assert_array_equal(
        np.sort(sequences, axis=1), np.broadcast_to(expected, sequences.shape)
    )


def ballot_ranking(ballot: RankBallot) -> tuple[frozenset[Candidate], ...]:
    assert ballot.ranking is not None
    return ballot.ranking


def compute_probability_of_ballot_by_slate(
    bloc: str, config: BlocSlateConfig, slate_ballot_type: Sequence[str]
):
    """
    Compute the probability of a ballot by slate given the sPL model.

    Args:
        bloc (str): The bloc for which to compute the probability of the ballot.
        config (BlocSlateConfig): The configuration for the bloc-slate type model.
        slate_ballot_type (Sequence[str]): The slate ballot type to compute the probability of.

    Returns:
        float: The probability of the ballot by slate.
    """

    num_cands_seen_per_slate = {slate: 0 for slate in config.slates}
    num_cands_per_slate = {
        slate: len(cand_list) for slate, cand_list in config.slate_to_candidates.items()
    }
    cohesion_values = config.cohesion_df.loc[bloc, list(config.slates)].to_numpy(dtype=float)
    prob_of_slate = {
        slate: float(value) for slate, value in zip(config.slates, cohesion_values, strict=True)
    }

    ballot_prob = 1.0
    for i, slate in enumerate(slate_ballot_type):
        ballot_prob *= prob_of_slate[slate]
        num_cands_seen_per_slate[slate] += 1

        if num_cands_seen_per_slate[slate] == num_cands_per_slate[slate]:
            del prob_of_slate[slate]
            prob_of_slate = {
                slate: prob / sum(prob_of_slate.values()) for slate, prob in prob_of_slate.items()
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


def test_SPL_completion_with_mixed_candidates(two_bloc_two_slate_mixed_config):
    config = two_bloc_two_slate_mixed_config
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
        match=(
            r"cohesion_df columns \(slates\) must be exactly \['Z', 'B'\] as defined in the "
            r"'slate_to_candidates' parameter. Got \['A', 'B'\]"
        ),
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
            ranking = ballot_ranking(ballot)
            if any(len(cand_set) > 1 for cand_set in ranking):
                raise ValueError(f"Tie occurred in ballot {ranking}")
            slate_ballot_type = tuple(
                [cand_to_slate_dict[cand] for cand_set in ranking for cand in cand_set]
            )
            slate_ballot_counts[slate_ballot_type] = (
                slate_ballot_counts.get(slate_ballot_type, 0) + ballot.weight
            )

        assert all(
            abs(
                slate_ballot_weight / profile.total_ballot_wt
                - compute_probability_of_ballot_by_slate(bloc, config, slate_ballot_type)
            )
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
                for cand_set in ballot_ranking(ballot)
                for cand in cand_set
                if isinstance(cand, str) and cand[0] == "A"
            )
            for ballot in profile.ballots
            for _ in range(int(ballot.weight))
        ]

        b_comparisons_profile = [
            tuple(
                cand
                for cand_set in ballot_ranking(ballot)
                for cand in cand_set
                if isinstance(cand, str) and cand[0] == "B"
            )
            for ballot in profile.ballots
            for _ in range(int(ballot.weight))
        ]

        assert (
            abs(
                a_comparisons_profile.count(("A1", "A2")) / profile.total_ballot_wt
                - config.preference_df.loc[bloc]["A1"]
                / (config.preference_df.loc[bloc]["A1"] + config.preference_df.loc[bloc]["A2"])
            )
            < PROB_THRESHOLD
        )

        assert (
            abs(
                b_comparisons_profile.count(("B1", "B2")) / profile.total_ballot_wt
                - config.preference_df.loc[bloc]["B1"]
                / (config.preference_df.loc[bloc]["B1"] + config.preference_df.loc[bloc]["B2"])
            )
            < PROB_THRESHOLD
        )


def test_one_bloc_three_slate_spl_distribution_matches_slate_ballot_dist(
    one_bloc_three_slate_config,
):
    config = one_bloc_three_slate_config

    profile = slate_pl_profile_generator(config, group_ballots=True)

    cand_to_slate_dict = {
        cand: slate for slate, cand_list in config.slate_to_candidates.items() for cand in cand_list
    }
    slate_ballot_counts = {}
    bloc = "X"

    for ballot in profile.ballots:
        ranking = ballot_ranking(ballot)
        if any(len(cand_set) > 1 for cand_set in ranking):
            raise ValueError(f"Tie occurred in ballot {ranking}")
        slate_ballot_type = tuple(
            [cand_to_slate_dict[cand] for cand_set in ranking for cand in cand_set]
        )
        slate_ballot_counts[slate_ballot_type] = (
            slate_ballot_counts.get(slate_ballot_type, 0) + ballot.weight
        )

    assert all(
        abs(
            slate_ballot_weight / profile.total_ballot_wt
            - compute_probability_of_ballot_by_slate(bloc, config, slate_ballot_type)
        )
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
                for cand_set in ballot_ranking(ballot)
                for cand in cand_set
                if isinstance(cand, str) and cand[0] == slate and cand[-1] in ["1", "2"]
            )
            for ballot in profile.ballots
            for _ in range(int(ballot.weight))
        ]

        assert (
            abs(
                cand_comparisons_profile.count((f"{slate}1", f"{slate}2")) / profile.total_ballot_wt
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
        allow_zero_support_candidates=True,
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
            for cand_set in ballot_ranking(ballot)
            for cand in cand_set
            if isinstance(cand, str) and cand[0] != "A"
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
        abs(prob - 1 / 3) < PROB_THRESHOLD for prob in zero_support_slate_perms_dist.values()
    )
