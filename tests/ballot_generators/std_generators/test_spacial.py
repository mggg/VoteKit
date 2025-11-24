import pytest
import numpy as np

from votekit.ballot_generator import (
    spacial_profile_and_positions_generator,
    clustered_spacial_profile_and_positions_generator,
    onedim_spacial_profile_generator,
)
from votekit.pref_profile import RankProfile


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


def test_1D_completion():
    profile = onedim_spacial_profile_generator(
        candidates=["W1", "W2", "C1", "C2"], number_of_ballots=100
    )
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100
