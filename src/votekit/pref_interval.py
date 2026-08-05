from __future__ import annotations

from types import MappingProxyType
from typing import Optional, Sequence

import numpy as np
from numpy.random import Generator

from votekit.types import Candidate


def combine_preference_intervals(
    intervals: list[PreferenceInterval],
    proportions: list[float],
    *,
    allow_zero_support: bool = False,
):
    """
    Combine a list of preference intervals given a list of proportions used to reweight each
    interval.

    Args:
        intervals (list[PreferenceInterval]): A list of PreferenceInterval objects to combine.
        proportions (list[float]): A list of floats used to reweight the PreferenceInterval objects.
            Proportion $i$ will reweight interval $i$.
        allow_zero_support (bool): If True, candidates with zero support are allowed in the
            combined interval. If False, all candidates must have strictly positive support.

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
        },
        allow_zero_support=allow_zero_support,
    )


class PreferenceInterval:
    """
    PreferenceInterval class, contains preference for individual candidates stored as relative
    share of the interval [0,1].

    Attributes:
        interval (dict): A dictionary representing the given PreferenceInterval.
            The keys are candidate names, and the values are floats representing that candidates
            share of the interval. Does not have to sum to one, the init method will renormalize.
        candidates (frozenset): A frozenset of candidates.

    Raises:
        ValueError: If support values cannot be normalized (sum to <= 0).
    """

    def __init__(self, interval: dict, *, allow_zero_support: bool = False):
        """
        Initializes a PreferenceInterval object.

        Args:
            interval (dict): A dictionary representing the given PreferenceInterval.
                The keys are candidate names, and the values are floats representing that candidates
                share of the interval. Does not have to sum to one, the init method will
                renormalize.
            allow_zero_support (bool): If True, candidates with zero support are allowed. If False,
                all candidates must have strictly positive support.
        """
        self.interval = MappingProxyType(interval)
        self.candidates = frozenset(self.interval.keys())
        self._allow_zero_support = allow_zero_support

        self._check_for_normalizable_interval()
        self._normalize()

    @classmethod
    def from_dirichlet(
        cls,
        candidates: Sequence[Candidate],
        alpha: float,
        *,
        allow_zero_support: bool = False,
        sort_strengths_descending: bool = False,
        rng_seed: Optional[int] = None,
        numpy_rng: Optional[Generator] = None,
    ):
        """
        Samples a PreferenceInterval from the Dirichlet distribution on the candidate simplex.
        Alpha tends to 0 is strong support, alpha tends to infinity is uniform support, alpha = 1
        is all bets are off.

        Args:
            candidates (Sequence[Candidate]): List of candidates.
                Candidates can be strings, integers, or mix of both.
            alpha (float): Alpha parameter for Dirichlet distribution.
            allow_zero_support (bool): If True, candidates with zero support are allowed. If False,
                all candidates must have strictly positive support.
            sort_strengths_descending (bool):
                If True, the candidates are assigned their support values in descending order
                according to the list passed to candidates.
                If False, the candidates are assigned support values in random order.
            rng_seed (int, optional)): seed for RNG, allows for reproducible results given the same
                inputs. Seed set to None by default, different results will be generated each time.
            numpy_rng (Generator, optional): NumPy random number generator. Pass a seeded instance
                for reproducible results; defaults to None for non-deterministic results.

        Returns:
            PreferenceInterval
        """
        if rng_seed is not None and numpy_rng is not None:
            raise ValueError("Cannot give a rng_seed and rng. Choose one.")
        numpy_rng = None
        if rng_seed is not None:
            numpy_rng = np.random.default_rng(seed=rng_seed)
        elif numpy_rng is not None:
            numpy_rng = numpy_rng
        else:
            numpy_rng = np.random.default_rng()

        probs = list(numpy_rng.dirichlet(alpha=[alpha] * len(candidates)))

        if not allow_zero_support:
            probs = [p + 10e-12 if p == 0 else p for p in probs]

        pref_interval = (
            {
                cand: strength
                for cand, strength in zip(candidates, sorted(probs, reverse=True), strict=True)
            }
            if sort_strengths_descending
            else {cand: strength for cand, strength in zip(candidates, probs, strict=True)}
        )

        return cls(
            pref_interval,
            allow_zero_support=allow_zero_support,
        )

    def _check_for_normalizable_interval(self):
        """
        Check if the interval can be normalized.

        Raises:
            ValueError: If support values sum to <= 0.
        """
        if self.interval and any(v < 0 for v in self.interval.values()):
            raise ValueError("Support values must be non-negative.")
        if self.interval and sum(self.interval.values()) <= 0:
            raise ValueError("Support values must sum to a positive number.")
        if not self._allow_zero_support:
            zero_support_cands = [c for c, s in self.interval.items() if s <= 0]
            if zero_support_cands:
                raise ValueError(
                    "Support values must be strictly positive for all candidates unless "
                    "allow_zero_support=True."
                )

    def _normalize(self):
        """
        Normalize a PreferenceInterval so the support values sum to 1.
        """
        summ = sum(self.interval.values())

        self.interval = MappingProxyType({c: s / summ for c, s in self.interval.items()})

    def __eq__(self, other):
        if not isinstance(other, PreferenceInterval):
            return False

        if not len(self.interval) == len(other.interval):
            return False

        else:
            # Round to 8 decimal places to avoid floating point issues
            return all(
                round(other.interval[key], 8) == round(value, 8)
                for key, value in self.interval.items()
            )

    def __repr__(self):
        printed_interval = {c: round(v, 4) for c, v in self.interval.items()}
        return str(printed_interval)
