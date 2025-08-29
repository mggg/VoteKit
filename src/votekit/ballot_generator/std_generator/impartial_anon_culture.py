import math
import numpy as np
import random
from typing import Dict
from collections import Counter

from votekit.pref_profile import PreferenceProfile
from votekit.utils import index_to_lexicographic_ballot, build_df_from_ballot_samples


class ImpartialAnonymousCulture:
    """
    Impartial Anonymous Culture model wher each profile is equally likely. Equivalent to the ballot
    simplex with an alpha value of 1.

    Args:

    Attributes:
    """

    def __init__(self, **data):
        if "candidates" not in data:  # and "slate_to_candidates" not in data:
            raise ValueError(
                "At least one of candidates or slate_to_candidates must be provided."
            )
        if "candidates" in data:
            self.candidates = data["candidates"]

        self._MAX_BINOM_EXPERIMENT_SIZE = 2**63 - 1

    def _total_ballots(self, n_candidates, max_ballot_length):
        return sum(
            math.comb(n_candidates, i) * math.factorial(i)
            for i in range(1, max_ballot_length + 1)
        )

    def generate_profile(
        self,
        number_of_ballots,
        max_ballot_length=None,
    ) -> PreferenceProfile | Dict:
        if max_ballot_length is None:
            max_ballot_length = len(self.candidates)

        return self._generate_profile_optimized(number_of_ballots, max_ballot_length)

    def _generate_profile_optimized(
        self, num_ballots: int, max_ballot_length: int
    ) -> PreferenceProfile | Dict:
        # choose index as sampled 0 to N, do this n! times
        num_cands = len(self.candidates)
        num_gaps = num_ballots  # + 1
        gap_freq = np.zeros(
            num_gaps, dtype=int
        )  # record the number of gaps in stars/bars

        # rather than iterate n! times, we perform multiple
        # multinomial experiments
        num_valid_ballots = self._total_ballots(num_cands, max_ballot_length)
        for _ in range(num_valid_ballots):
            gap_freq[random.randint(0, num_gaps) - 1] += 1

        # TODO: Double check possible off by 1 errors here and in `gap_freq`
        ballot_indices = np.cumsum(gap_freq) - 1
        ballots_as_cand_ind = [
            tuple(
                index_to_lexicographic_ballot(
                    ballot_ind,
                    num_cands,
                    max_ballot_length,
                    self._total_ballots,
                    always_use_total_valid_ballots_method=False,
                )
            )
            for ballot_ind in ballot_indices
        ]
        ballots_as_counter = Counter(ballots_as_cand_ind)
        pp_df = build_df_from_ballot_samples(dict(ballots_as_counter), self.candidates)
        pp_df.index.name = "Ballot Index"
        return PreferenceProfile(
            df=pp_df,
            contains_rankings=True,
            max_ranking_length=len(self.candidates),
            candidates=self.candidates,
        )
