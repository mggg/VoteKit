from votekit.profile import PreferenceProfile
from metrics.distances import earth_mover_dist, Lp_dist
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
import sys

sys.path.append("../")


# Help function for MDS Plot
def distance_matrix(
    pp_set1: set[PreferenceProfile],
    pp_set2: set[PreferenceProfile],
    distance: str,
    ballot_graph,
):
    """
    creates pairwise distance matrix between preference profiles. The i-th and j-th entry are
    pairwise distance between i-th col preference profile and the j-th row preference profile. \n
    pp_set1 and pp_set2: set of preference profiles \n
    distance: Use "L1" for L1 dist, "earth mover" for earth mover dist.
    """
    rows = len(pp_set1)
    cols = len(pp_set2)
    dist_matrix = np.zeros((rows, cols))
    pp_arr1 = list(pp_set1)
    pp_arr2 = list(pp_set2)

    if distance == "L1":
        for i in range(rows):
            for j in range(cols):
                if i != j:
                    dist_matrix[i][j] = Lp_dist(pp_arr1[i], pp_arr2[j], 1)
        return dist_matrix

    if distance == "earth mover":
        for i in range(rows):
            for j in range(cols):
                if i != j:
                    dist_matrix[i][j] = earth_mover_dist(
                        pp_arr1[i], pp_arr2[j], ballot_graph
                    )
        return dist_matrix
    else:
        raise ValueError("Unsupported distance type")


def plot_MDS(
    pp_set1: set[PreferenceProfile],
    pp_set2: set[PreferenceProfile],
    distance: str,
    ballot_graph,
):
    """
    Creates a multidimensional scaling plot \n
    distance: Use "L1" for L1 distance, "earth mover" for earth mover distance. \n
    ballot graph: in the future earth mover dist will compute this so this will be removed
    """
    dist_matrix = distance_matrix(pp_set1, pp_set2, distance, ballot_graph)
    mds = manifold.MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        dissimilarity="precomputed",
        n_jobs=1,
        normalized_stress="auto",
    )
    pos = mds.fit(np.array(dist_matrix)).embedding_
    num_rows, num_cols = np.shape(dist_matrix)
    plt.scatter(pos[0:num_rows, 0], pos[0:num_cols, 1], color="b", lw=0, s=5)
    plt.title("MDS Plot for Pair Wise Election Distances")
    plt.show()
    return plt
