from votekit.profile import PreferenceProfile
from votekit.graphs.models import BallotGraph
import numpy as np
import ot  # type: ignore
import networkx as nx  # type: ignore
from typing import Union, Optional


def earth_mover_dist(pp1: PreferenceProfile, pp2: PreferenceProfile) -> int:
    """
    Computes the earth mover distance between two elections. \n
    Assumes both elections share the same candidates
    """
    # create ballot graph
    graph = BallotGraph(source=pp2, complete=True)
    ballot_graph = graph.from_profile(profile=pp2, complete=True)

    # Solving Earth Mover Distance
    electA_distr = np.array(em_array(pp=pp1))
    electB_distr = np.array(em_array(pp=pp2))

    # Floyd Warshall Shortest Distance alorithm. Returns a dictionary of shortest path for each node
    fw_dist_dict = nx.floyd_warshall(ballot_graph)
    keys_list = sorted(fw_dist_dict.keys())
    cost_matrix = np.zeros((len(keys_list), len(keys_list)))
    for i in range(len(keys_list)):
        node_dict = fw_dist_dict[keys_list[i]]
        cost_col = [value for key, value in sorted(node_dict.items())]
        cost_matrix[i] = cost_col
    earth_mover_matrix = ot.emd(electA_distr, electB_distr, cost_matrix)

    # Hadamard Product = Earth mover dist between two matrices
    earth_mover_dist = np.sum(np.multiply(cost_matrix, earth_mover_matrix))
    return earth_mover_dist


def lp_dist(
    pp1: PreferenceProfile,
    pp2: PreferenceProfile,
    p_value: Optional[Union[int, str]] = 1,
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


# helper functions
# these functions comvert a list of preference profiles into distribution arrays
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


def em_array(pp: PreferenceProfile):
    ballot_graph = BallotGraph(source=pp)
    node_cand_map = ballot_graph.label_cands(sorted(pp.get_candidates()))
    pp_dict = pp.to_dict(True)

    # invert node_cand_map to map to pp_dict
    inverted = {v: k for k, v in node_cand_map.items()}
    combined_dict = {k: 0 for k in node_cand_map}

    # map nodes with weight of corresponding rank
    node_pp_dict = {inverted[key]: pp_dict[key] for key in pp_dict}

    complete_election_dict = combined_dict | node_pp_dict
    elect_distr = [
        float(complete_election_dict[key])
        for key in sorted(complete_election_dict.keys())
    ]

    return elect_distr
