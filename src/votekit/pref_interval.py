from __future__ import annotations
import numpy as np
import types


def combine_preference_intervals(
    intervals: list[PreferenceInterval], proportions: list[float]
):
    """
    Combine a list of preference intervals given a list of proportions used to reweight each
    interval.

    Args:
        intervals (list[PreferenceInterval]): A list of PreferenceInterval objects to combine.
        proportions (list[float]): A list of floats used to reweight the PreferenceInterval objects.
            Proportion $i$ will reweight interval $i$.

    Returns:
        PreferenceInterval: A combined PreferenceInterval object.

    Raises:
        ValueError: If the intervals have disjoint candidate sets.
        ValueError: If the proportions do not sum to 1.
    """
    if not (
        len(frozenset.union(*[pi.candidates for pi in intervals]))
        == sum(len(pi.candidates) for pi in intervals)
    ):
        raise ValueError("Intervals must have disjoint candidate sets")

    if round(sum(proportions), 8) != 1:
        raise ValueError("Proportions must sum to 1.")

    return PreferenceInterval(
        interval={
            key: value * prop
            for pi, prop in zip(intervals, proportions)
            for key, value in pi.interval.items()
        }
    )


class PreferenceInterval:
    """
    PreferenceInterval class, contains preference for individual candidates stored as relative
    share of the interval [0,1].

    Args:
        interval (dict): A dictionary representing the given PreferenceInterval.
            The keys are candidate names, and the values are floats representing that candidates
            share of the interval. Does not have to sum to one, the init method will renormalize.
            Includes candidates with zero support.

    Attributes:
        interval (dict): A dictionary representing the given PreferenceInterval.
            The keys are candidate names, and the values are floats representing that candidates
            share of the interval. Does not have to sum to one, the init method will renormalize.
            Does not include candidates with zero support.
        candidates (frozenset): A frozenset of candidates.

    Raises:
        ValueError: If there are candidates with zero support.
    """

    def __init__(self, interval: dict):
        self.interval = types.MappingProxyType(interval)
        self.candidates = frozenset(self.interval.keys())

        self._check_for_zero_support_cands()
        self._normalize()

    @classmethod
    def from_dirichlet(cls, candidates: list[str], alpha: float):
        """
        Samples a PreferenceInterval from the Dirichlet distribution on the candidate simplex.
        Alpha tends to 0 is strong support, alpha tends to infinity is uniform support, alpha = 1
        is all bets are off.

        Args:
            candidates (list): List of candidate strings.
            alpha (float): Alpha parameter for Dirichlet distribution.


        Returns:
            PreferenceInterval
        """
        probs = list(np.random.default_rng().dirichlet(alpha=[alpha] * len(candidates)))
        probs = [p + 10e-12 if p == 0 else p for p in probs]

        return cls({c: s for c, s in zip(candidates, probs)})

    def _check_for_zero_support_cands(self):
        """
        Check for candidates with zero support in the interval.

        Raises:
            ValueError: If there are candidates with zero support.
        """
        for cand, value in self.interval.items():
            if value == 0:
                raise ValueError(f"Candidate {cand} has zero support.")

    def _normalize(self):
        """
        Normalize a PreferenceInterval so the support values sum to 1.
        """
        summ = sum(self.interval.values())

        self.interval = types.MappingProxyType(
            {c: s / summ for c, s in self.interval.items()}
        )

    def __eq__(self, other):
        if not isinstance(other, PreferenceInterval):
            return False

        if not len(self.interval) == len(other.interval):
            return False

        else:
            return all(
                round(other.interval[key], 8) == round(value, 8)
                for key, value in self.interval.items()
            )

    def __repr__(self):
        printed_interval = {c: round(v, 4) for c, v in self.interval.items()}
        return str(printed_interval)
