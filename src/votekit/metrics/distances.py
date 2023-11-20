from votekit.pref_profile import PreferenceProfile
from votekit.graphs.ballot_graph import BallotGraph
import numpy as np
import ot  # type: ignore
import networkx as nx  # type: ignore
from typing import Union, Optional


def earth_mover_dist(pp1: PreferenceProfile, pp2: PreferenceProfile) -> int:
    """
    Computes the earth mover distance between two elections.
    Assumes both elections share the same candidates.

    Args:
        pp1: PreferenceProfile for first election.
        pp2: PreferenceProfile for second election.

    Returns:
        Earth mover distance between inputted elections.
    """
    # create ballot graph
    ballot_graph = BallotGraph(source=pp2).graph
    # ballot_graph = graph.from_profile(profile=pp2, complete=True)

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
    Use 'inf' for infinity norm.
    Assumes both elections share the same candidates.

    Args:
        pp1: PreferenceProfile for first election.
        pp2: PreferenceProfile for second election.
        p_value: Distance parameter, 1 for Manhattan, 2 for Euclidean 
            or 'inf' for Chebyshev distance.

    Returns:
        Lp distance between two elections.
    """
    pp_list = [pp1, pp2]
    pp_2arry = profiles_to_ndarrys(pp_list)
    electA = pp_2arry[:, 0]
    electB = pp_2arry[:, 1]

    if isinstance(p_value, int):
        sum = 0
        for i in range(len(electA)):
            diff = (abs(electA[i] - electB[i])) ** p_value
            sum += diff
        lp_dist = sum ** (1 / p_value)
        return lp_dist

    elif p_value == "inf":
        diff = [abs(x - y) for x, y in zip(electA, electB)]
        return max(diff)

    else:
        raise ValueError("Unsupported input type")


# helper functions
# these functions comvert a list of preference profiles into distribution arrays
def profiles_to_ndarrys(profiles: list[PreferenceProfile]):
    """
    Converts a list of PreferenceProfile into an ndarray,
    a matrix like object. The cols represent each profile, 
    rows are the cast ballots, and each element represents
    the frequency a ballot type occurs for a PreferenceProfile.
    Each column will sum to one since weights are standardized.
    This is usefule for computing election Lp distances between
    elections.

    Args:
        profiles (list[PreferenceProfile])
        : a list of PreferenceProfiles

    Returns:
    An ndarray.
    """
    cast_ballots: list = []
    profile_dicts: list[dict] = []
    for pp in profiles:
        election_dict = pp.to_dict(standardize=True)
        profile_dicts.append(election_dict)
        for key in election_dict.keys():
            if key not in cast_ballots:
                cast_ballots.append(key)
    combined_dict = {ranking: 0 for ranking in cast_ballots}
    rows = len(cast_ballots)
    cols = len(profile_dicts)
    electn_ndarry = np.zeros((rows, cols))

    for i in range(len(profile_dicts)):
        election = combined_dict | profile_dicts[i]
        elect_distr = [float(election[key]) for key in sorted(election.keys())]
        electn_ndarry[:, i] = elect_distr
    return electn_ndarry


def em_array(pp: PreferenceProfile) -> list:
    """
    Converts a PreferenceProfile into a distribution using ballot graphs.

    Args:
        pp: PreferenceProfile for a given election.

    Returns:
        Distribution of ballots for an election.
    """
    ballot_graph = BallotGraph(source=pp)
    node_cand_map = ballot_graph.label_cands(sorted(pp.get_candidates()))
    pp_dict = pp.to_dict(True)

    # invert node_cand_map to map to pp_dict
    # split is used to remove the custom labeling from the ballotgraph
    inverted = {v.split(":")[0]: k for k, v in node_cand_map.items()}
    combined_dict = {k: 0 for k in node_cand_map}

    # map nodes with weight of corresponding rank
    # labels on ballotgraph are strings so need to convert key to string
    node_pp_dict = {inverted[str(key)]: pp_dict[key] for key in pp_dict}

    complete_election_dict = combined_dict | node_pp_dict
    elect_distr = [
        float(complete_election_dict[key])
        for key in sorted(complete_election_dict.keys())
    ]

    return elect_distr
