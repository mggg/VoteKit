import pytest

from votekit.ballot_generator import (
    name_PlackettLuce,
)

from votekit.pref_interval import PreferenceInterval


def test_not_cand_and_not_slate_to_cand():
    blocs = {"R": 0.7, "D": 0.3}
    cohesion = {"R": {"R": 0.7, "D": 0.3}, "D": {"D": 0.6, "R": 0.4}}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}

    with pytest.raises(ValueError):
        name_PlackettLuce(
            bloc_voter_prop=blocs,
            cohesion_parameters=cohesion,
            alphas=alphas,
        )


def test_all_nec_params():
    # missing bloc voter prop
    # bloc_voter_prop={"W": 0.7, "C": 0.3},
    with pytest.raises(ValueError):
        name_PlackettLuce(
            candidates=["W1", "W2", "C1", "C2"],
            pref_intervals_by_bloc={
                "W": {
                    "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                    "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
                },
                "C": {
                    "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                    "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
                },
            },
            cohesion_parameters={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
        )

    # missing pref_intervals
    with pytest.raises(ValueError):
        name_PlackettLuce(
            candidates=["W1", "W2", "C1", "C2"],
            bloc_voter_prop={"W": 0.7, "C": 0.3},
            cohesion_parameters={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
        )


def test_incorrect_bloc_props():
    # should sum to 1
    blocs = {"R": 0.7, "D": 0.4}

    cohesion = {"R": {"R": 0.7, "D": 0.3}, "D": {"D": 0.6, "R": 0.4}}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}
    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}

    with pytest.raises(ValueError):
        name_PlackettLuce.from_params(
            slate_to_candidates=slate_to_cands,
            bloc_voter_prop=blocs,
            cohesion_parameters=cohesion,
            alphas=alphas,
        )
