from votekit.plots import compute_MDS, plot_MDS  # , plot_summary_stats
from votekit.ballot_generator import name_PlackettLuce
from votekit.metrics import lp_dist
import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt


def test_compute_MDS():
    bloc_prop = {"R": 0.7, "D": 0.3}
    cohesion = {"R": {"R": 0.7, "D": 0.3}, "D": {"D": 0.6, "R": 0.4}}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}
    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}

    pl = name_PlackettLuce.from_params(
        slate_to_candidates=slate_to_cands,
        bloc_voter_prop=bloc_prop,
        cohesion_parameters=cohesion,
        alphas=alphas,
    )

    data = {"PL": [pl.generate_profile(number_of_ballots=10) for _ in range(10)]}

    coord_dict = compute_MDS(data, lp_dist)

    assert isinstance(coord_dict, dict)
    assert list(coord_dict.keys()) == list(data.keys())
    assert isinstance(coord_dict["PL"], tuple)
    assert len(coord_dict["PL"][0]) == 10 and len(coord_dict["PL"][1]) == 10


def test_plot_MDS():
    bloc_prop = {"R": 0.7, "D": 0.3}
    cohesion = {"R": {"R": 0.7, "D": 0.3}, "D": {"D": 0.6, "R": 0.4}}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}
    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}

    pl = name_PlackettLuce.from_params(
        slate_to_candidates=slate_to_cands,
        bloc_voter_prop=bloc_prop,
        cohesion_parameters=cohesion,
        alphas=alphas,
    )

    data = {"PL": [pl.generate_profile(number_of_ballots=10) for _ in range(10)]}

    coord_dict = compute_MDS(data, lp_dist)

    ax = plot_MDS(coord_dict)

    assert isinstance(ax, Axes)
    plt.close()


def test_seed_MDS():
    bloc_prop = {"R": 0.7, "D": 0.3}
    cohesion = {"R": {"R": 0.7, "D": 0.3}, "D": {"D": 0.6, "R": 0.4}}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}
    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}

    pl = name_PlackettLuce.from_params(
        slate_to_candidates=slate_to_cands,
        bloc_voter_prop=bloc_prop,
        cohesion_parameters=cohesion,
        alphas=alphas,
    )

    data = {"PL": [pl.generate_profile(number_of_ballots=10) for _ in range(10)]}

    coord_dict_1 = compute_MDS(data, lp_dist, random_seed=10)
    coord_dict_2 = compute_MDS(data, lp_dist, random_seed=10)
    coord_dict_3 = compute_MDS(data, lp_dist, random_seed=15)

    assert np.array_equal(coord_dict_1["PL"][0], coord_dict_2["PL"][0])
    assert np.array_equal(coord_dict_1["PL"][1], coord_dict_2["PL"][1])
    assert not np.array_equal(coord_dict_3["PL"][0], coord_dict_2["PL"][0])
    assert not np.array_equal(coord_dict_3["PL"][1], coord_dict_2["PL"][1])
