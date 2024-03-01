from votekit.pref_profile import PreferenceProfile
from typing import Callable
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from typing import Dict, Optional
from sklearn import manifold  # type: ignore


# Helper function for MDS Plot
def distance_matrix(
    pp_arr: list[PreferenceProfile], distance: Callable[..., int], *args, **kwargs
):
    """
    Creates pairwise distance matrix between PreferenceProfile. The $(i,j)$ entry is the pairwise
    distance between $i$th and the $j$th PreferenceProfile.

    Args:
        pp_arr: List of PreferenceProfiles.
        distance: Callable distance function type. See distances.py in the metrics module.

    Returns:
        dist_matrix (ndarray): Distance matrix for an election.
    """
    rows = len(pp_arr)
    dist_matrix = np.zeros((rows, rows))

    for i in range(rows):
        for j in range(i + 1, rows):
            dist_matrix[i][j] = distance(pp_arr[i], pp_arr[j], *args, **kwargs)
            dist_matrix[j][i] = dist_matrix[i][j]
    return dist_matrix


def compute_MDS(
    data: Dict[str, list[PreferenceProfile]],
    distance: Callable[..., int],
    random_seed: int = 47,
    *args,
    **kwargs
):
    """
    Computes the coordinates of an MDS plot. This is time intensive, so it is decoupled from
    `plot_mds` to allow users to flexibly use the coordinates.

    Args:
        data: Dictionary with key being a string label and value being list of
                    PreferenceProfiles. ex: {'PL with alpha = 4': list[PreferenceProfile]}
        distance: Distance function. See distance.py.
        random_seed (int): an integer seed to allow for reproducible MDS plots. Defaults to 47.

    Returns:
        coord_dict (dict): a dictionary whose keys match `data` and whose values are tuples of
        numpy arrays (x_list, y_list) of coordinates for the MDS plot.
    """
    # combine all lists to create distance matrix
    combined_pp = []
    for pp_list in data.values():
        combined_pp.extend(pp_list)

    # compute distance matrix
    dist_matrix = distance_matrix(combined_pp, distance, *args, **kwargs)

    mds = manifold.MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        dissimilarity="precomputed",
        n_jobs=1,
        normalized_stress="auto",
        random_state=random_seed,
    )
    pos = mds.fit(np.array(dist_matrix)).embedding_

    coord_dict = {}
    start_pos = 0
    for key, value_list in data.items():
        # color, label, marker = key
        end_pos = start_pos + len(value_list)
        coord_dict[key] = (pos[start_pos:end_pos, 0], pos[start_pos:end_pos, 1])
        start_pos += len(value_list)

    return coord_dict


def plot_MDS(
    coord_dict: dict,
    plot_kwarg_dict: Optional[dict] = None,
    legend: bool = True,
    title: bool = True,
):
    """
    Creates an MDS plot from the output of `compute_MDS` with legend labels matching the keys
    of `coord_dict`.

    Args:
        coord_dict: Dictionary with key being a string label and value being tuple
        (x_list, y_list), coordinates for the MDS plot.
        Should be piped in from `compute_MDS`.

        plot_kwarg_dict: Dictionary with keys matching coord_dict and values are kwarg dictionaries
            that will be passed to matplotlib `scatter`.

        legend: boolean for plotting the legend. Defaults to True.

        title: boolean for plotting the title. Defaults to True.

    Returns:
        fig: a matplotlib fig
    """

    # Plot data
    fig, ax = plt.subplots()

    for key, value in coord_dict.items():
        x, y = value
        if plot_kwarg_dict and key in plot_kwarg_dict:
            ax.scatter(x, y, label=key, **plot_kwarg_dict[key])
        else:
            ax.scatter(x, y, label=key)

    if title:
        ax.set_title("MDS Plot for Pairwise Election Distances")
    if legend:
        ax.legend()

    ax.set_aspect("equal")

    return fig
