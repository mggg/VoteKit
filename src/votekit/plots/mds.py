from votekit.pref_profile import PreferenceProfile
from typing import Optional, Callable
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from typing import Dict
from sklearn import manifold  # type: ignore

# Helper function for MDS Plot
def distance_matrix(
    pp_arr: list[PreferenceProfile], distance: Callable[..., int], *args, **kwargs
):
    """
    Creates pairwise distance matrix between preference profiles. The i-th and
    j-th entry are pairwise distance between i-th col preference profile and
    the j-th row preference profile.

    Args:
        pp_arr: List of preference profiles
        distance: Callable distance function type. See distance.py

    Returns:
        Distance matrix for an election
    """
    rows = len(pp_arr)
    dist_matrix = np.zeros((rows, rows))

    for i in range(rows):
        for j in range(i + 1, rows):
            dist_matrix[i][j] = distance(pp_arr[i], pp_arr[j], *args, **kwargs)
            dist_matrix[j][i] = dist_matrix[i][j]
    return dist_matrix


def plot_MDS(
    data: Dict[str, list[PreferenceProfile]],
    distance: Callable[..., int],
    marker_size: Optional[int] = 5,
    *args,
    **kwargs
):
    """
    Creates a multidimensional scaling plot.

    Args:
        data: Dictionary with key being a 'color' and value being list of \n
        PreferenceProfiles. ex: {'color': list[PreferenceProfile]}
        distance: Distance function
        marker_size: Size of plotted points

    Returns:
        An MDS plot
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
    )
    pos = mds.fit(np.array(dist_matrix)).embedding_

    # Plot and color data
    start_pos = 0
    for key, value_list in data.items():
        end_pos = start_pos + len(value_list)
        plt.scatter(
            pos[start_pos:end_pos, 0],
            pos[start_pos:end_pos, 1],
            color=key,
            lw=0,
            s=marker_size,
        )
        start_pos += len(value_list)
    plt.title("MDS Plot for Pair Wise Election Distances")
    plt.show()
    return plt
