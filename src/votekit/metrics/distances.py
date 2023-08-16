from votekit.profile import PreferenceProfile
import numpy as np
import ot
import networkx as nx
from typing import Union, Optional

# TODO: Update em_dist to manually create graph in the function. Graph should be able to handle
# or incomplete ballost


def earth_mover_dist(
    pp1: PreferenceProfile, pp2: PreferenceProfile, ballot_graph
) -> int:
    """
    Computes the earth mover distance between two elections. \n
    Assumes both elections share the same candidates
    """
    # Solving Earth Mover Distance
    election_arrays = profilePairs_to_arrays(pp1, pp2)
    electA_distr = np.array(election_arrays[0])
    electB_distr = np.array(election_arrays[1])

    # TODO:create ballot graph:

    # Floyd Warshall Shortest Distance alorithm. Returns a dictionary of shortest path for each node
    fw_dist_dict = nx.floyd_warshall(ballot_graph)
    keys_list = sorted(fw_dist_dict.keys())
    cost_matrix = np.zeros((len(keys_list), len(keys_list)))
    for i in range(len(keys_list)):
        node_dict = fw_dist_dict[keys_list[i]]
        cost_col = [value for key, value in sorted(node_dict.items())]
        cost_matrix[i] = cost_col
    # print('cost matrix shape',cost_matrix.shape) 
    count = 0 
    if  (electA_distr.shape[0] == cost_matrix.shape[0]) & (electB_distr.shape[0]== cost_matrix.shape[1]):
        count += 1
    if count == 0:
        print(electA_distr, electB_distr)
    # print(electA_distr.shape[0], cost_matrix.shape[0])
    # print(electB_distr.shape[0], cost_matrix.shape[1])
    
    earth_mover_matrix = ot.emd(electA_distr, electB_distr, cost_matrix)

    # Hadamard Product = Earth mover dist between two matrices
    earth_mover_dist = np.sum(np.multiply(cost_matrix, earth_mover_matrix))
    return earth_mover_dist


def lp_dist(
    pp1: PreferenceProfile, pp2: PreferenceProfile, p_value: Optional[Union[int, str]] = 1
) -> int:
    """
    Computes the L_p distance between two election distributions.
    Use 'inf' for infinity norm. \n
    Assumes both elections share the same candidates.
    """
    election_arrays = profilePairs_to_arrays(pp1, pp2)
    electA_distr = np.array(election_arrays[0])
    electB_distr = np.array(election_arrays[1])

    if isinstance(p_value, int):
        sum = 0
        for i in range(len(electA_distr)):
            diff = (abs(electA_distr[i] - electB_distr[i])) ** p_value
            sum += diff
        lp_dist = sum ** (1 / p_value)
        return lp_dist

    elif p_value == "inf":
        diff = [x - y for x, y in zip(electA_distr, electB_distr)]
        return max(diff)

    else:
        raise ValueError("Unsupported input type")


def profilePairs_to_arrays(
    pp1: PreferenceProfile, pp2: PreferenceProfile
) -> tuple[list[float], list[float]]:
    """
    Converts two elections i.e preference profiles into distribution arrays.\n
    This is useful to compute distance between two elections
    """
    elect1 = pp1.to_dict(standardize=True)
    elect2 = pp2.to_dict(standardize=True)
    all_rankings = set(elect1.keys()).union(elect2.keys())
    combined_dict = {key: 0 for key in all_rankings}

    elect1 = combined_dict | elect1
    elect2 = combined_dict | elect2

    electA_distr = [float(elect1[key]) for key in sorted(elect1.keys())]
    electB_distr = [float(elect2[key]) for key in sorted(elect2.keys())]
    return electA_distr, electB_distr
