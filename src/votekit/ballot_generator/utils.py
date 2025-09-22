import itertools as it
import random
import numpy as np
from typing import Union, Mapping
import psutil


def system_memory() -> dict[str, float]:
    """
    Returns a dictionary with system memory information in GiB via psutil.

    Returns:
        dict[str, float]: A dictionary with keys 'total_gib', 'available_gib',
            'used_gib', and 'percent' representing the total, available, and used
            memory in GiB and the percentage of used memory.
    """
    vm = psutil.virtual_memory()
    return {
        "total_gib": vm.total / 2**30,
        "available_gib": vm.available / 2**30,
        "used_gib": vm.used / 2**30,
        "percent": vm.percent,
    }


# TODO: Fix this up to be more readable. Also make sure to mention keys of
# cohesion_parameters_for_bloc are slates now.
def sample_cohesion_ballot_types(
    slate_to_non_zero_candidates: dict[str, list[str]],
    num_ballots: int,
    cohesion_parameters_for_bloc: Mapping[str, Union[float, int]],
) -> list[list[str]]:
    """
    Returns a list of ballots; each ballot is a list of slate names (strings)
    in the order they appear on that ballot.

    Args:
        slate_to_non_zero_candidates (dict[str, list[str]]):
        num_ballots (int):
        cohesion_parameters_for_bloc (Mapping[str, Union[float, int]]):

    Returns:
        list[list[str]]: A list of lists, where each list contains the bloc names in the order
            they appear on the ballot.
    """
    candidates = list(it.chain.from_iterable(slate_to_non_zero_candidates.values()))

    ballots: list[list[str]] = [[] for _ in range(num_ballots)]

    coin_flips = list(np.random.uniform(size=len(candidates) * num_ballots))

    def which_bin(dist_bins: list[float], flip: float) -> int:
        for i, left in enumerate(dist_bins[:-1]):
            if left < flip <= dist_bins[i + 1]:
                return i
        return len(dist_bins) - 2

    blocs_og, values_og = [list(x) for x in zip(*cohesion_parameters_for_bloc.items())]

    for j in range(num_ballots):
        blocs = blocs_og.copy()
        values = values_og.copy()

        distribution_bins: list[float] = [0.0] + [
            sum(values[: i + 1]) for i in range(len(blocs))
        ]
        ballot_type: list[str] = [""] * len(candidates)

        for i, flip in enumerate(
            coin_flips[j * len(candidates) : (j + 1) * len(candidates)]
        ):
            bloc_index = which_bin(distribution_bins, float(flip))
            bloc_type = blocs[bloc_index]
            ballot_type[i] = bloc_type

            if ballot_type.count(bloc_type) == len(
                slate_to_non_zero_candidates[bloc_type]
            ):
                del blocs[bloc_index]
                del values[bloc_index]
                total_value_sum = sum(values)

                if total_value_sum == 0 and len(values) > 0:
                    # remaining blocs have zero cohesion → fill with random permutation
                    remaining_blocs = [
                        b
                        for b in blocs
                        for _ in range(len(slate_to_non_zero_candidates[b]))
                    ]
                    random.shuffle(remaining_blocs)
                    ballot_type[i + 1 :] = remaining_blocs
                    break

                # renormalize and recompute bins
                values = [v / total_value_sum for v in values]
                distribution_bins = [0.0] + [
                    sum(values[: k + 1]) for k in range(len(blocs))
                ]

        ballots[j] = ballot_type

    return ballots
