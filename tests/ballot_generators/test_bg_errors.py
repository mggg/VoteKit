import pytest
import numpy as np

from votekit.ballot_generator import (
    cambridge_profile_generator,
    spacial_profile_and_positions_generator,
    clustered_spacial_profile_and_positions_generator,
    BlocSlateConfig,
)


from votekit.pref_interval import PreferenceInterval


def test_Cambridge_maj_bloc_error():
    # need to provide both W_bloc and C_bloc
    with pytest.raises(ValueError):
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
        cambridge_profile_generator(config, majority_bloc="A")

    # must be distinct
    with pytest.raises(ValueError):
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
        cambridge_profile_generator(config, majority_bloc="A", minority_bloc="A")


def test_spatial_generator():
    candidates = [str(i) for i in range(25)]
    uniform_params = {"low": 0, "high": 1, "size": 2}
    normal_params = {"loc": 0.5, "scale": 0.1, "size": 2}

    def bad_dist(x, y, z):
        return x + y + z

    with pytest.raises(TypeError, match="Invalid kwargs for the voter distribution."):
        spacial_profile_and_positions_generator(
            number_of_ballots=1,
            candidates=candidates,
            voter_dist=np.random.normal,
            voter_dist_kwargs=uniform_params,
            candidate_dist=np.random.normal,
            candidate_dist_kwargs=normal_params,
        )

    with pytest.raises(
        TypeError, match="Invalid kwargs for the candidate distribution."
    ):
        spacial_profile_and_positions_generator(
            number_of_ballots=1,
            candidates=candidates,
            voter_dist=np.random.normal,
            voter_dist_kwargs=normal_params,
            candidate_dist=np.random.normal,
            candidate_dist_kwargs=uniform_params,
        )

    with pytest.raises(
        TypeError,
        match="Distance function is invalid or "
        "incompatible with voter/candidate distributions.",
    ):
        spacial_profile_and_positions_generator(
            number_of_ballots=1,
            candidates=candidates,
            voter_dist=np.random.normal,
            voter_dist_kwargs=normal_params,
            candidate_dist=np.random.normal,
            candidate_dist_kwargs=normal_params,
            distance=bad_dist,
        )


def test_clustered_spatial_generator():
    candidates = [str(i) for i in range(25)]
    uniform_params = {"low": 0, "high": 1, "size": 2}
    normal_params = {"loc": 0.5, "scale": 0.1, "size": 2}

    def bad_dist(x, y, z):
        return x + y + z

    with pytest.raises(TypeError, match="Invalid kwargs for the voter distribution."):
        clustered_spacial_profile_and_positions_generator(
            number_of_ballots=1,
            candidates=candidates,
            voter_dist=np.random.normal,
            voter_dist_kwargs=uniform_params,
            candidate_dist=np.random.normal,
            candidate_dist_kwargs=normal_params,
        )

    with pytest.raises(
        TypeError, match="Invalid kwargs for the candidate distribution."
    ):
        clustered_spacial_profile_and_positions_generator(
            number_of_ballots=1,
            candidates=candidates,
            voter_dist=np.random.normal,
            voter_dist_kwargs=normal_params,
            candidate_dist=np.random.normal,
            candidate_dist_kwargs=uniform_params,
        )

    with pytest.raises(
        TypeError,
        match="Distance function is invalid or "
        "incompatible with voter/candidate distributions.",
    ):
        clustered_spacial_profile_and_positions_generator(
            number_of_ballots=1,
            candidates=candidates,
            voter_dist=np.random.normal,
            voter_dist_kwargs=normal_params,
            candidate_dist=np.random.normal,
            candidate_dist_kwargs=normal_params,
            distance=bad_dist,
        )

    with pytest.raises(ValueError, match="Input voter distribution not supported."):
        clustered_spacial_profile_and_positions_generator(
            number_of_ballots=1,
            candidates=candidates,
            voter_dist=np.random.uniform,
            voter_dist_kwargs=normal_params,
            candidate_dist=np.random.normal,
            candidate_dist_kwargs=normal_params,
        )


# FIX: Get this method up and running again
# def test_MCMC_subsample_chain_length_error():
#     """
#     Test to check chain_length < num_ballots to be sampled
#     """
#
#     # Initialization of parameters; doesn't matter for this test
#     candidates = ["W1", "W2", "C1", "C2"]
#     pref_intervals_by_bloc = {
#         "W": {
#             "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
#             "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
#         },
#         "C": {
#             "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
#             "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
#         },
#     }
#     bloc_voter_prop = {"W": 0.7, "C": 0.3}
#     cohesion_parameters = {"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.6, "W": 0.4}}
#     bloc_voter_prop = {"W": 0.7, "C": 0.3}
#
#     bt = name_BradleyTerry(
#         candidates=candidates,
#         pref_intervals_by_bloc=pref_intervals_by_bloc,
#         bloc_voter_prop=bloc_voter_prop,
#         cohesion_parameters=cohesion_parameters,
#     )
#
#     number_of_ballots = 3001
#
#     # Error where minority bloc needs 0.3*10,000 = 3,000 number of ballots, where 10,000 is the preset chain_length
#     with pytest.raises(
#         ValueError,
#         match="The number of ballots to be sampled is more than the chain length; supply a greater chain length.",
#     ):
#         bt.generate_profile_MCMC_even_subsample(number_of_ballots=number_of_ballots)
#
#     chain_length = 100
#     number_of_ballots = 101
#
#     with pytest.raises(
#         ValueError,
#         match="The number of ballots to be sampled is more than the chain length; supply a greater chain length.",
#     ):
#         bt.generate_profile_MCMC_even_subsample(
#             number_of_ballots=number_of_ballots, chain_length=chain_length
#         )
