import math
import numpy as np
import random
from typing import Dict
from collections import Counter

from votekit.pref_profile import PreferenceProfile
from votekit.utils import index_to_lexicographic_ballot, build_df_from_ballot_samples


class ImpartialCulture:
    """
    Impartial Culture model where each ballot is equally likely.
    Equivalent to the ballot simplex with an alpha value of infinity.

    Args:
        **data: kwargs to be passed to ``BallotGenerator`` parent class.

    Attributes:
        alpha (float): Alpha parameter for Dirichlet distribution.
    """

    _total_ballots_cache: dict[tuple[int, int], int] = {}

    def __init__(self, **data):
        if "candidates" not in data:  # and "slate_to_candidates" not in data:
            raise ValueError(
                "At least one of candidates or slate_to_candidates must be provided."
            )
        if "candidates" in data:
            self.candidates = data["candidates"]

        use_ballots_cache_key = "use_total_ballots_cache"
        self._use_total_ballots_cache = data.get(use_ballots_cache_key, True)

    def _clear_cache(self):
        ImpartialCulture._total_ballots_cache = {}

    def generate_profile(
        self,
        number_of_ballots: int,
        max_ballot_length=None,
        allow_short_ballots=False,
    ) -> PreferenceProfile | Dict:
        if max_ballot_length is None:
            max_ballot_length = len(self.candidates)
        elif max_ballot_length > len(self.candidates):
            raise Exception("Max ballot length larger than number of candidates given.")

        if allow_short_ballots:
            return self._generate_profile_optimized_with_short(
                number_of_ballots, max_ballot_length
            )
        else:
            return self._generate_profile_optimized_non_short(
                number_of_ballots, max_ballot_length
            )

    def _generate_profile_optimized_non_short(
        self, number_of_ballots: int, ballot_length: int
    ):
        """
        Generate an IC preference profile using Fisher-Yates shuffle
        {number_of_ballots} times. Used to generate a profile when
        short ballots are disallowed

        args:
            number_of_ballots: int; the number of ballots to generate
        returns:
            PreferenceProfile
        """
        num_cands = len(self.candidates)
        ballots_as_ind = [
            tuple(np.random.choice(num_cands, size=ballot_length, replace=False))
            for _ in range(number_of_ballots)
        ]
        ballots_as_counter = Counter(ballots_as_ind)
        pp_df = build_df_from_ballot_samples(dict(ballots_as_counter), self.candidates)
        pp_df.index.name = "Ballot Index"
        return PreferenceProfile(
            df=pp_df,
            contains_rankings=True,
            max_ranking_length=len(self.candidates),
            candidates=self.candidates,
        )

    def _generate_profile_optimized_with_short(
        self, number_of_ballots: int, max_ballot_length: int = -1
    ) -> PreferenceProfile | Dict:
        """
        Generate an IC profile in the case where short ballots are
        allowed. Randomly sample indices between 0 and number_of_valid
        ballots, we do this {number_of_ballots} times. Then we convert
        the indices to ballots using a help function

        args:
            number_of_ballots: the number of ballots to generate for
                the profile
            max_ballot_length: the maximum length allowed in the
            profile
        returns:
            PreferenceProfile
        """
        num_cands = len(self.candidates)
        if max_ballot_length == -1:
            max_ballot_length = num_cands
        total_ballots = self._total_ballots(num_cands, max_ballot_length)

        # sample indices (representing allowed ballots) uniformally at
        # random
        ballot_inds = [
            random.randint(0, total_ballots - 1) for _ in range(number_of_ballots)
        ]
        ballots_as_cand_ind = [
            tuple(
                index_to_lexicographic_ballot(
                    ballot_ind, num_cands, max_ballot_length, self._total_ballots
                )
            )
            for ballot_ind in ballot_inds
        ]

        # Instantiate the preference profile using a dataframe
        ballots_as_counter = Counter(ballots_as_cand_ind)
        pp_df = build_df_from_ballot_samples(dict(ballots_as_counter), self.candidates)
        pp_df.index.name = "Ballot Index"
        return PreferenceProfile(
            df=pp_df,
            contains_rankings=True,
            max_ranking_length=len(self.candidates),
            candidates=self.candidates,
        )

    def _total_ballots(self, n_candidates, max_ballot_length):
        if not self._use_total_ballots_cache:
            return sum(
                math.comb(n_candidates, i) * math.factorial(i)
                for i in range(1, max_ballot_length + 1)
            )

        key = (n_candidates, max_ballot_length)
        if key not in ImpartialCulture._total_ballots_cache:
            ImpartialCulture._total_ballots_cache[key] = sum(
                math.comb(n_candidates, i) * math.factorial(i)
                for i in range(1, max_ballot_length + 1)
            )
        return ImpartialCulture._total_ballots_cache[key]
