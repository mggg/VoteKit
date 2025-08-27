import itertools as it
import numpy as np
import random


def sample_cohesion_ballot_types(
    slate_to_non_zero_candidates: dict,
    num_ballots: int,
    cohesion_parameters_for_bloc: dict,
):
    """
    Used to generate bloc orderings given cohesion parameters.

    Args:
        slate_to_non_zero_candidates (dict): A mapping of slates to their list of non_zero
                                            candidates.
        num_ballots (int): the number of ballots to generate.
        cohesion_parameters_for_bloc (dict): A mapping of blocs to cohesion parameters.
                                Note, this is equivalent to one value in the cohesion_parameters
                                dictionary.


    Returns:
      A list of lists of length `num_ballots`, where each sub-list contains the bloc names in order
      they appear on that ballot.
    """
    candidates = list(it.chain(*list(slate_to_non_zero_candidates.values())))
    ballots = [[-1]] * num_ballots
    # pre-compute coin flips
    coin_flips = list(np.random.uniform(size=len(candidates) * num_ballots))

    def which_bin(dist_bins, flip):
        for i, bin in enumerate(dist_bins):
            if bin < flip <= dist_bins[i + 1]:
                return i

    blocs_og, values_og = [list(x) for x in zip(*cohesion_parameters_for_bloc.items())]

    for j in range(num_ballots):
        blocs, values = blocs_og.copy(), values_og.copy()
        # Pre-calculate distribution_bins
        distribution_bins = [0] + [sum(values[: i + 1]) for i in range(len(blocs))]
        ballot_type = [-1] * len(candidates)

        for i, flip in enumerate(
            coin_flips[j * len(candidates) : (j + 1) * len(candidates)]
        ):
            bloc_index = which_bin(distribution_bins, flip)
            bloc_type = blocs[bloc_index]
            ballot_type[i] = bloc_type

            # Check if adding candidate exhausts a slate of candidates
            if ballot_type.count(bloc_type) == len(
                slate_to_non_zero_candidates[bloc_type]
            ):
                del blocs[bloc_index]
                del values[bloc_index]
                total_value_sum = sum(values)

                if total_value_sum == 0 and len(values) > 0:
                    # this indicates that remaining blocs have 0 cohesion with this bloc
                    # so complete ballot with random permutation of remaining blocs
                    remaining_blocs = [
                        b
                        for b in blocs
                        for _ in range(len(slate_to_non_zero_candidates[b]))
                    ]
                    random.shuffle(remaining_blocs)
                    ballot_type[i + 1 :] = remaining_blocs
                    break

                values = [v / total_value_sum for v in values]
                distribution_bins = [0] + [
                    sum(values[: i + 1]) for i in range(len(blocs))
                ]

        ballots[j] = ballot_type

    return ballots
