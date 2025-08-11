from votekit.pref_profile import PreferenceProfile, profile_to_ranking_dict
from votekit.graphs.ballot_graph import BallotGraph
import numpy as np
import networkx as nx  # type: ignore
from typing import Union, Optional
from scipy.optimize import linprog
from scipy.sparse import identity, kron, vstack, csr_matrix
from scipy.sparse.csgraph import floyd_warshall


def emd_via_scipy_linear_program(
    source_distribution: np.ndarray,
    target_distribution: np.ndarray,
    cost_matrix: np.ndarray,
) -> float:
    """
    Compute the Earth Mover's Distance (EMD) between two discrete distributions using
    a linear programming formulation of the optimal transport problem.

    Classical Formulation:
        minimize ⟨cost_matrix, transport_plan⟩
        subject to:
            transport_plan @ 1 = source_distribution
            transport_plan.T @ 1 = target_distribution
            transport_plan >= 0

    Linear Program:
        Suppose that 'a' is the source distribution, 'b' is the target distribution, C is the
        cost matrix, and 'X' is the transport plan. Let 'x' be the row-major vectorization of 'X'
        and let 'c' be the vectorization of 'C'. The linear program can be expressed as:

        Then let 'A' be the constraint matrix formed by the Kronecker products of identity
        matrices and the ones vector, so

        A = [I_n ⊗  1_m^T, 1_n^T ⊗ I_m]

        where I_n is the identity matrix of size n (number of sources) and I_m is the identity
        matrix of size m (number of targets). The linear program can be expressed as:

        minimize c^T x

        subject to:
            Ax = y

        where 'y' is the concatenation of the source and target distributions y = [a, b]^T.

    Args:
        source_distribution (np.ndarray): Mass distribution of the source (length n).
        target_distribution (np.ndarray): Mass distribution of the target (length m).
        cost_matrix (np.ndarray): n×m matrix of transportation costs.

    Returns:
        float: The computed Earth Mover's Distance (minimum transport cost).
    """
    nonzero_source_mask = source_distribution > 0
    nonzero_target_mask = target_distribution > 0

    if not nonzero_source_mask.any() and not nonzero_target_mask.any():
        return 0.0

    # Trim to only nonzero entries to reduce LP size
    trimmed_source = source_distribution[nonzero_source_mask]
    trimmed_target = target_distribution[nonzero_target_mask]

    # NOTE: np.ix_ is used to create a meshgrid for indexing so you grab the correct submatrix
    trimmed_cost_matrix = cost_matrix[np.ix_(nonzero_source_mask, nonzero_target_mask)]
    num_sources, num_targets = trimmed_cost_matrix.shape

    # For each source, we need to ensure that the total mass is fully received by the targets
    # Suppose n=2 sources, m=3 targets, and transport plan:
    #
    #     X = [[x11, x12, x13],
    #          [x21, x22, x23]]
    #
    # Flattened row-major:
    #     x = [x11, x12, x13, x21, x22, x23]   # length n*m = 6
    #
    # Now with the Kronecker product, we get the constraints:
    # Row constraints: I_n ⊗ [1 1 1]   → sums each source's row:
    #     [[1 1 1 0 0 0],               # x11 + x12 + x13 = a1
    #      [0 0 0 1 1 1]]               # x21 + x22 + x23 = a2
    #
    # Col constraints: [1 1] ⊗ I_m     → sums each target's column:
    #     [[1 0 0 1 0 0],               # x11 + x21 = b1
    #      [0 1 0 0 1 0],               # x12 + x22 = b2
    #      [0 0 1 0 0 1]]               # x13 + x23 = b3
    row_constraints = kron(
        identity(num_sources, format="csr"),
        csr_matrix(np.ones((1, num_targets))),
    )
    col_constraints = kron(
        csr_matrix(np.ones((1, num_sources))),
        identity(num_targets, format="csr"),
    )

    # Matrix formulation of the constraints:
    A_eq = vstack([row_constraints, col_constraints], format="csr")

    # NOTE: np.r_ is used to concatenate the trimmed source and target distributions
    b_eq = np.r_[trimmed_source, trimmed_target]

    # Solve LP
    result = linprog(
        trimmed_cost_matrix.ravel(),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0, None),
        method="highs",
    )

    if not result.success:
        raise RuntimeError(f"linprog failed: {result.message}")

    return result.fun


def earth_mover_dist(pp1: PreferenceProfile, pp2: PreferenceProfile) -> float:
    """
    Computes the Earth Mover's Distance (EMD) between two preference profiles.
    Assumes both elections share the same candidates.

    Args:
        pp1 (PreferenceProfile): PreferenceProfile for first profile.
        pp2 (PreferenceProfile): PreferenceProfile for second profile.

    Returns:
        float: Earth Mover's Distance between two profiles.
    """
    ballot_graphA = BallotGraph(source=pp1, fix_short=True)
    ballot_graphB = BallotGraph(source=pp2, fix_short=True)

    assert set(ballot_graphA.graph.nodes) == set(ballot_graphB.graph.nodes)

    electA_distr = np.array(list(ballot_graphA.node_weights.values()))
    electA_distr /= np.sum(electA_distr)
    electB_distr = np.array(list(ballot_graphB.node_weights.values()))
    electB_distr /= np.sum(electB_distr)

    A = nx.to_scipy_sparse_array(ballot_graphB.graph, weight="weight", dtype=float)
    cost_matrix = floyd_warshall(A)

    return emd_via_scipy_linear_program(
        source_distribution=electA_distr,
        target_distribution=electB_distr,
        cost_matrix=cost_matrix,
    )


def lp_dist(
    pp1: PreferenceProfile,
    pp2: PreferenceProfile,
    p_value: Optional[Union[int, str]] = 1,
) -> int:
    r"""
    Computes the :math:`L_p` distance between two profiles.
    Use 'inf' for infinity norm.
    Assumes both elections share the same candidates.

    Args:
        pp1 (PreferenceProfile): PreferenceProfile for first profile.
        pp2 (PreferenceProfile): PreferenceProfile for second profile.
        p_value (Union[int, str], optional): :math:`L_p` distance parameter. Use "inf" for
            :math:`\infty`.

    Returns:
        int: :math:`L_p` distance between two profiles.
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


def profiles_to_ndarrys(profiles: list[PreferenceProfile]):
    """
    Converts a list of preference profiles into an ndarray. The columns
    represent each profile, rows are the cast ballots, and each element represents
    the frequency a ballot type occurs for a ``PreferenceProfile``. Each column will sum to one
    since weights are standardized. This is useful for computing election :math:`L_p` distances
    between profiles.

    Args:
        profiles (list[PreferenceProfile]): A list of PreferenceProfiles.

    Returns:
        numpy.ndarray: computed matrix of ballot frequencies.
    """
    cast_ballots: list = []
    profile_dicts: list[dict] = []
    for pp in profiles:
        election_dict = profile_to_ranking_dict(pp, standardize=True)
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


def euclidean_dist(point1: np.ndarray, point2: np.ndarray) -> float:
    return float(np.linalg.norm(point1 - point2, ord=2))
