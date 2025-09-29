from votekit.pref_profile import RankProfile, rank_profile_to_ranking_dict
import numpy as np
from typing import Union, Optional, Sequence
from scipy.optimize import linprog
from scipy.sparse import identity, kron, vstack, csr_matrix
from scipy.stats import kendalltau


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
        Suppose that 'a' is the source distribution, 'b' is the target distribution, 'C' is the
        cost matrix, and 'X' is the transport plan. Let 'x' be the row-major vectorization of 'X'
        and let 'c' be the vectorization of 'C'.

        Then let 'A' be the constraint matrix formed by the Kronecker products of identity
        matrices and the ones vector, so

        A = [I_n ⊗  1_m^T, 1_n^T ⊗ I_m]

        where 'I_n' is the identity matrix of size 'n' (number of sources) and 'I_m' is the identity
        matrix of size 'm' (number of targets). The linear program can be expressed as:

        minimize c^T x

        subject to:
            Ax = y

        where 'y' is the concatenation of the source and target distributions 'y = [a, b]^T'.

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

    if (source_distribution < 0).any() or (target_distribution < 0).any():
        raise ValueError(
            "Negative entries in source or target distributions are not allowed."
        )

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

    ret = result.fun
    if ret is None:
        raise RuntimeError("linprog returned None as the optimal value.")

    assert ret is not None
    return ret


def __vaildate_ranking_distance_inputs(
    ranking1: Sequence[int], ranking2: Sequence[int], n_candidates: int
) -> tuple[set[int], set[int], set[int]]:
    """
    Validates the inputs for computing the distance between two rankings.

    Args:
        ranking1 (Sequence[int]): The first ranking as a sequence of candidate indices.
        ranking2 (Sequence[int]): The second ranking as a sequence of candidate indices.
        n_candidates (int): The total number of candidates in the rankings.

    Returns:
        tuple[set[int], set[int], set[int]]: A tuple containing three sets:
            - The set of candidates in the first ranking.
            - The set of candidates in the second ranking.
            - The union of both sets, representing all candidates in both rankings.

    Raises:
        ValueError: If the rankings contain duplicates or if the number of candidates jointly
            contained within the rankings exceeds the total number of candidates.
    """
    ranked1_set = set(ranking1)
    ranked2_set = set(ranking2)
    full_ranking_set = ranked1_set.union(ranked2_set)

    if len(ranked1_set) != len(ranking1):
        raise ValueError(
            f"The ranking {ranking1} contains duplicates and is not suitable for distance "
            "computation."
        )
    if len(ranked2_set) != len(ranking2):
        raise ValueError(
            f"The ranking {ranking2} contains duplicates and is not suitable for distance "
            "computation."
        )
    if len(full_ranking_set) > n_candidates:
        raise ValueError(
            "The number of candidates in the rankings exceeds the total number of candidates."
        )
    return ranked1_set, ranked2_set, full_ranking_set


def __compute_bubble_sort_distance(
    ranking1: Sequence[int], ranking2: Sequence[int], full_ranking_set: set[int]
) -> float:
    r"""
    Computes the bubble sort distance between two rankings.

    This function makes use of the Kendall tau correlation coefficient to compute the
    bubble sort distance, which is defined as the number of adjacent transpositions needed to
    transform one ranking into another. The formula for the Kendall tau statistic is given by:

    .. math::
        \tau = \frac{2(C - D}{n(n-1)}

    where :math:`C` is the number of concordant pairs, :math:`C` is the number of discordant pairs,
    and :math:`n` is the number of items being ranked. The bubble sort distance, :math:`D` is then
    computed as:

    .. math::
        \text{Bubble Sort Distance} = \frac{n(n-1)}{4} \cdot (1 - \tau)

    where :math:`n` is the number of candidates in the full ranking set.

    Args:
        ranking1 (Sequence[int]): The first ranking as a sequence of candidate indices.
        ranking2 (Sequence[int]): The second ranking as a sequence of candidate indices.
        full_ranking_set (set[int]): The set of all candidates in both rankings.

    Returns:
        float: The computed bubble sort distance between the two rankings.
    """
    if len(ranking1) == 0 or len(ranking2) == 0:
        return 0.0

    full_ranking_1 = list(ranking1) + [c for c in ranking2 if c not in set(ranking1)]
    full_ranking_2 = list(ranking2) + [c for c in ranking1 if c not in set(ranking2)]

    assert set(full_ranking_1) == set(full_ranking_2) == full_ranking_set

    ranking1_sort_idx = np.argsort(full_ranking_1)
    ranking2_sort_idx = np.argsort(full_ranking_2)

    tau, _ = kendalltau(ranking1_sort_idx, ranking2_sort_idx)
    full_ranking_count = len(full_ranking_set)
    t_value = (full_ranking_count * (full_ranking_count - 1)) / 2
    return int(round(t_value * (1 - float(tau)) / 2))  # type: ignore[return-value]


def compute_ranking_distance_on_ballot_graph(
    ranking1: Sequence[int],
    ranking2: Sequence[int],
    n_candidates: int,
):
    """
    Computes the distance between two rankings on a ballot graph.

    Args:
        ranking1 (Sequence[int]): The first ranking as a sequence of candidate indices.
        ranking2 (Sequence[int]): The second ranking as a sequence of candidate indices.
        n_candidates (int): The number of candidates in the rankings.

    Returns:
        float: The computed distance between the two rankings.
    """
    if n_candidates <= 0:
        raise ValueError("The number of candidates must be greater than zero.")
    if n_candidates == 1:
        return 0.0
    if ranking1 == ranking2:
        return 0.0

    ranked1_set, ranked2_set, full_ranking_set = __vaildate_ranking_distance_inputs(
        ranking1, ranking2, n_candidates
    )

    bubble_sort_distance = __compute_bubble_sort_distance(
        ranking1, ranking2, full_ranking_set
    )

    insertion_deletion_set = ranked1_set.symmetric_difference(ranked2_set)
    insertion_deletion_distance = len(insertion_deletion_set) / 2

    if (
        ranked1_set.intersection(ranked2_set) == set()
        and len(ranking1) > 0
        and len(ranking2) > 0
    ):
        # Case where there might an insertion and a deletion at the end. Both are free, so we
        # credit them back 0.5 each.
        insertion_credit = 1 if len(full_ranking_set) == n_candidates else 0.0
    else:
        # Case where there might be a single insertion or deletion at the end which we need to
        # credit back.
        insertion_credit = 0.5 if len(full_ranking_set) == n_candidates else 0.0

    # Case were where two ballots equivalent to full rankings swap the last two candidates
    if (
        len(full_ranking_set) == n_candidates
        and len(ranking1) != 0
        and len(ranking2) != 0
        and ranking1[:-2] == ranking2[:-2]
    ):
        insertion_deletion_distance = 0.0
        insertion_credit = 0.0

    return bubble_sort_distance + insertion_deletion_distance - insertion_credit


def __build_simultaneous_profile_distribution(
    pp1: RankProfile, pp2: RankProfile
) -> dict[tuple[int, ...], tuple[float, float]]:
    """
    Builds a simultaneous distribution of two preference profiles, where each key is a tuple
    representing a ranking and the value is a tuple containing the weights from each of the
    profiles in order. If the ranking is not present in one of the profiles, the weight is set to
    0.0.

    Args:
        pp1 (RankProfile): RankProfile for first profile.
        pp2 (RankProfile): RankProfile for second profile.

    Returns:
        dict[tuple[int, ...], tuple[float, float]]: A dictionary where keys are tuples of candidate
            indices representing rankings and values are tuples of weights from each profile
            in the order of pp1 and pp2.
    """
    profile1 = pp1.group_ballots()
    profile2 = pp2.group_ballots()

    cand_to_index_mapping = {
        cand: i for i, cand in enumerate(sorted(profile1.candidates))
    }

    profile_distribution_dict = dict()

    assert profile1.max_ranking_length is not None
    profile1_ranking_array = profile1.df[
        [f"Ranking_{i}" for i in range(1, profile1.max_ranking_length + 1)]
    ].to_numpy()
    profile1_wt_vector = profile1.df["Weight"].astype(float).to_numpy()
    profile1_wt_vector = profile1_wt_vector / np.sum(profile1_wt_vector)

    if (profile1_ranking_array == frozenset({})).any():
        raise ValueError(
            "The first profile contains an empty ranking, which is not allowed."
        )

    for idx, ranking_tuple in enumerate(profile1_ranking_array):
        tup = tuple(
            [
                cand_to_index_mapping[cand]
                for cand_set in ranking_tuple
                for cand in cand_set
                if cand != "~"
            ]
        )

        profile_distribution_dict[tup] = (float(profile1_wt_vector[idx]), 0.0)

    assert profile2.max_ranking_length is not None
    profile2_ranking_array = profile2.df[
        [f"Ranking_{i}" for i in range(1, profile2.max_ranking_length + 1)]
    ].to_numpy()
    profile2_wt_vector = profile2.df["Weight"].astype(float).to_numpy()
    profile2_wt_vector = profile2_wt_vector / np.sum(profile2_wt_vector)

    if (profile2_ranking_array == frozenset({})).any():
        raise ValueError(
            "The second profile contains an empty ranking, which is not allowed."
        )

    for idx, ranking_tuple in enumerate(profile2_ranking_array):
        tup = tuple(
            [
                cand_to_index_mapping[cand]
                for cand_set in ranking_tuple
                for cand in cand_set
                if cand != "~"
            ]
        )

        profile_distribution_dict[tup] = (
            profile_distribution_dict.get(tup, (0.0, 0.0))[0],
            float(profile2_wt_vector[idx]),
        )

    return profile_distribution_dict


def earth_mover_dist(pp1: RankProfile, pp2: RankProfile) -> float:
    """
    Computes the Earth Mover's Distance (EMD) between two preference profiles.
    Assumes both elections share the same candidates.

    Args:
        pp1 (RankProfile): RankProfile for first profile.
        pp2 (RankProfile): RankProfile for second profile.

    Returns:
        float: Earth Mover's Distance between two profiles.
    """
    if set(pp1.candidates) != set(pp2.candidates):
        raise ValueError("The two profiles must have the same candidates.")

    if not isinstance(pp1, RankProfile) or not isinstance(pp2, RankProfile):
        raise ValueError(
            "Both profiles must contain rankings to compute the Earth Mover's Distance."
        )
    if pp1.max_ranking_length != pp2.max_ranking_length:
        raise ValueError(
            "Both profiles must have the same maximum ranking length to compute the Earth Mover's "
            "Distance."
        )

    profile_distribution_dict = __build_simultaneous_profile_distribution(pp1, pp2)

    profile1_distribution = np.zeros(len(profile_distribution_dict))
    profile2_distribution = np.zeros(len(profile_distribution_dict))
    for idx, (v1, v2) in enumerate(profile_distribution_dict.values()):
        profile1_distribution[idx] = v1
        profile2_distribution[idx] = v2

    distribution_len = len(profile_distribution_dict)
    cost_matrix = np.zeros((distribution_len, distribution_len))

    n_candidates = len(pp1.candidates)
    all_ranking_tuples = list(profile_distribution_dict.keys())
    for idx1, rank1 in enumerate(all_ranking_tuples):
        for idx2 in range(idx1 + 1, len(all_ranking_tuples)):
            rank2 = all_ranking_tuples[idx2]
            cost_matrix[idx1, idx2] = compute_ranking_distance_on_ballot_graph(
                rank1, rank2, n_candidates
            )
            cost_matrix[idx2, idx1] = cost_matrix[idx1, idx2]

    return emd_via_scipy_linear_program(
        source_distribution=profile1_distribution,
        target_distribution=profile2_distribution,
        cost_matrix=cost_matrix,
    )


def lp_dist(
    pp1: RankProfile,
    pp2: RankProfile,
    p_value: Optional[Union[int, str]] = 1,
) -> int:
    r"""
    Computes the :math:`L_p` distance between two profiles.
    Use 'inf' for infinity norm.
    Assumes both elections share the same candidates.

    Args:
        pp1 (RankProfile): RankProfile for first profile.
        pp2 (RankProfile): RankProfile for second profile.
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


def profiles_to_ndarrys(profiles: list[RankProfile]):
    """
    Converts a list of preference profiles into an ndarray. The columns
    represent each profile, rows are the cast ballots, and each element represents
    the frequency a ballot type occurs for a ``RankProfile``. Each column will sum to one
    since weights are standardized. This is useful for computing election :math:`L_p` distances
    between profiles.

    Args:
        profiles (list[RankProfile]): A list of PreferenceProfiles.

    Returns:
        numpy.ndarray: computed matrix of ballot frequencies.
    """
    cast_ballots: list = []
    profile_dicts: list[dict] = []
    for pp in profiles:
        election_dict = rank_profile_to_ranking_dict(pp, standardize=True)
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
