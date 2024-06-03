from __future__ import annotations
import numpy as np
import types


def combine_preference_intervals(
    intervals: list[PreferenceInterval], proportions: list[float]
):
    """
        Combine a list of preference intervals given a list of proportions used to reweight each
        interval.

    **Arguments**
    `intervals`
    : list.  A list of PreferenceInterval objects to combine.

    `proportions`
    : list. A list of floats used to reweight the PreferenceInterval objects. Proportion $i$ will
    reweight interval $i$.
    """
    if not (
        len(frozenset.union(*[pi.candidates for pi in intervals]))
        == sum(len(pi.candidates) for pi in intervals)
    ):
        raise ValueError("Intervals must have disjoint candidate sets")

    if round(sum(proportions), 8) != 1:
        raise ValueError("Proportions must sum to 1.")

    sum_pi = PreferenceInterval(
        interval={
            key: value * prop
            for pi, prop in zip(intervals, proportions)
            for key, value in pi.interval.items()
        }
    )

    # carry along the candidates with zero support
    zero_cands = frozenset.union(*[pi.zero_cands for pi in intervals])

    # need to union to ensure that if one of the proportions is 0 those candidates are saved
    sum_pi.zero_cands = sum_pi.zero_cands.union(zero_cands)
    return sum_pi


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
        candidates (frozenset): A frozenset of candidates (with zero and non-zero support).
        non_zero_cands (frozenset): A frozenset of candidates with non-zero support.
        zero_cands (frozenset): A frozenset of candidates with zero support.
    """

    # TODO frozendict, frozenclass

    def __init__(self, interval: dict):
        self.interval = types.MappingProxyType(interval)
        self.candidates = frozenset(self.interval.keys())

        self.zero_cands: frozenset = frozenset()
        self.non_zero_cands: frozenset = frozenset()
        self._remove_zero_support_cands()
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

        return cls({c: s for c, s in zip(candidates, probs)})

    def _normalize(self):
        """
        Normalize a PreferenceInterval so the support values sum to 1.
        """
        summ = sum(self.interval.values())

        if summ == 0:
            raise ZeroDivisionError("There are no candidates with non-zero support.")

        self.interval = types.MappingProxyType(
            {c: s / summ for c, s in self.interval.items()}
        )

    def _remove_zero_support_cands(self):
        """
        Remove candidates with zero support from the interval. Store candidates
        with zero support as a set in the attribute `zero_cands`.

        Should only be run once.
        """

        if not self.zero_cands and not self.non_zero_cands:
            self.zero_cands = frozenset([c for c, s in self.interval.items() if s == 0])
            self.interval = types.MappingProxyType(
                {c: s for c, s in self.interval.items() if s > 0}
            )
            self.non_zero_cands = frozenset(self.interval.keys())

    def __eq__(self, other):
        if not isinstance(other, PreferenceInterval):
            raise TypeError("Both types must be PreferenceInterval.")

        if not self.zero_cands == other.zero_cands:
            return False

        if not self.non_zero_cands == other.non_zero_cands:
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
