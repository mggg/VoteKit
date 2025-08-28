import itertools as it
import numpy as np
import random
import warnings
from typing import Union, Tuple
import apportionment.methods as apportion

from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
from votekit.pref_interval import combine_preference_intervals, PreferenceInterval
from votekit.ballot_generator import BallotGenerator


class name_BradleyTerry(BallotGenerator):
    """
    Class for generating ballots using a name-BradleyTerry model. The probability of sampling
    the ranking :math:`X>Y>Z` is proportional to :math:`P(X>Y)*P(X>Z)*P(Y>Z)`.
    These individual probabilities are based on the preference interval: :math: `P(X>Y) = x/(x+y)`.
    Can be initialized with an interval or can be constructed with the Dirichlet distribution using
    the ``from_params`` method of ``BallotGenerator``.

    Args:
        slate_to_candidates (dict): Dictionary whose keys are bloc names and whose
            values are lists of candidate strings that make up the slate.
        bloc_voter_prop (dict): Dictionary whose keys are bloc strings and values are floats
                denoting population share.
        pref_intervals_by_bloc (dict): Dictionary whose keys are bloc strings and values are
            dictionaries whose keys are bloc strings and values are ``PreferenceInterval`` objects.
        cohesion_parameters (dict): Dictionary mapping of bloc string to dictionary whose
            keys are bloc strings and values are cohesion parameters,
            eg. ``{'bloc_1': {'bloc_1': .7, 'bloc_2': .2, 'bloc_3':.1}}``

    Attributes:
        candidates (list): List of candidate strings.
        slate_to_candidates (dict): Dictionary whose keys are bloc names and whose
            values are lists of candidate strings that make up the slate.
        bloc_voter_prop (dict): Dictionary whose keys are bloc strings and values are floats
                denoting population share.
        pref_intervals_by_bloc (dict): Dictionary whose keys are bloc strings and values are
            dictionaries whose keys are bloc strings and values are ``PreferenceInterval`` objects.
        pref_interval_by_bloc (dict): Dictionary whose keys are bloc strings and values are
            ``PreferenceInterval`` objects. This is constructed by rescaling the intervals
            from ``pref_intervals_by_bloc`` via the ``cohesion_parameters`` and concatenating them.
        cohesion_parameters (dict): Dictionary mapping of bloc string to dictionary whose
            keys are bloc strings and values are cohesion parameters,
            eg. ``{'bloc_1': {'bloc_1': .7, 'bloc_2': .2, 'bloc_3':.1}}``
    """

    def __init__(self, cohesion_parameters: dict, **data):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(cohesion_parameters=cohesion_parameters, **data)

        # if dictionary of pref intervals
        if isinstance(
            list(self.pref_intervals_by_bloc.values())[0], PreferenceInterval
        ):
            self.pref_interval_by_bloc = self.pref_intervals_by_bloc

        # if nested dictionary of pref intervals, combine by cohesion
        else:
            self.pref_interval_by_bloc = {
                bloc: combine_preference_intervals(
                    [self.pref_intervals_by_bloc[bloc][b] for b in self.blocs],
                    [self.cohesion_parameters[bloc][b] for b in self.blocs],
                )
                for bloc in self.blocs
            }

        if len(self.candidates) < 12:
            # precompute pdfs for sampling
            self.pdfs_by_bloc = {
                bloc: self._BT_pdf(self.pref_interval_by_bloc[bloc].interval)
                for bloc in self.blocs
            }
        else:
            warnings.warn(
                "For 12 or more candidates, exact sampling is computationally infeasible. \
                    Please only use the built in generate_profile_MCMC method."
            )

    def _calc_prob(self, permutations: list[tuple], cand_support_dict: dict) -> dict:
        """
        given a list of (possibly incomplete) rankings and the preference interval, \
        calculates the probability of observing each ranking

        Args:
            permutations (list[tuple]): a list of permuted rankings
            cand_support_dict (dict): a mapping from candidate to their \
            support (preference interval)

        Returns:
            dict: a mapping of the rankings to their probability
        """
        ranking_to_prob = {}
        for ranking in permutations:
            prob = 1
            for i in range(len(ranking)):
                cand_i = ranking[i]
                greater_cand_support = cand_support_dict[cand_i]
                for j in range(i + 1, len(ranking)):
                    cand_j = ranking[j]
                    cand_support = cand_support_dict[cand_j]
                    prob *= greater_cand_support / (greater_cand_support + cand_support)
            ranking_to_prob[ranking] = prob
        return ranking_to_prob

    def _make_pow(self, lst):
        """
        Helper method for _BT_pdf.
        Takes is a list representing the preference lengths of each candidate
        in a permutation.
        Computes the numerator of BT probability.
        """
        ret = 1
        m = len(lst)
        for i, val in enumerate(lst):
            if i < m - 1:
                ret *= val ** (m - i - 1)
        return ret

    def _BT_pdf(self, dct):
        """
        Construct the BT pdf as a dictionary (ballot, probability) given a preference
        interval as a dictionary (candidate, preference).
        """

        # gives PI lengths for each candidate in permutation
        def pull_perm(lst):
            nonlocal dct
            return [dct[i] for i in lst]

        new_dct = {
            perm: self._make_pow(pull_perm(perm))
            for perm in it.permutations(dct.keys(), len(dct))
        }
        summ = sum(new_dct.values())
        return {key: value / summ for key, value in new_dct.items()}

    def generate_profile(
        self, number_of_ballots, by_bloc: bool = False
    ) -> Union[PreferenceProfile, Tuple]:
        """
        Args:
            number_of_ballots (int): The number of ballots to generate.
            by_bloc (bool): True if you want the generated profiles returned as a tuple
                ``(pp_by_bloc, pp)``, where ``pp_by_bloc`` is a dictionary with keys = bloc strings
                and values = ``PreferenceProfile`` and ``pp`` is the aggregated profile. False if
                you only want the aggregated profile. Defaults to False.

        Returns:
            Union[PreferenceProfile, Tuple]
        """
        # the number of ballots per bloc is determined by Huntington-Hill apportionment

        bloc_props = list(self.bloc_voter_prop.values())
        ballots_per_block = dict(
            zip(
                self.blocs,
                apportion.compute("huntington", bloc_props, number_of_ballots),
            )
        )

        pp_by_bloc = {b: PreferenceProfile() for b in self.blocs}

        for bloc in self.blocs:
            num_ballots = ballots_per_block[bloc]

            # Directly initialize the list using good memory trick
            ballot_pool = [Ballot()] * num_ballots
            zero_cands = self.pref_interval_by_bloc[bloc].zero_cands
            pdf_dict = self.pdfs_by_bloc[bloc]

            # Directly use the keys and values from the dictionary for sampling
            rankings, probs = zip(*pdf_dict.items())

            # The return of this will be a numpy array, so we don't need to make it into a list
            sampled_indices = np.array(
                np.random.choice(
                    a=len(rankings),
                    size=num_ballots,
                    p=probs,
                ),
                ndmin=1,
            )

            for j, index in enumerate(sampled_indices):
                ranking = [frozenset({cand}) for cand in rankings[index]]

                # Add any zero candidates as ties only if they exist
                if zero_cands:
                    ranking.append(frozenset(zero_cands))

                ballot_pool[j] = Ballot(ranking=tuple(ranking), weight=1)

            pp = PreferenceProfile(ballots=tuple(ballot_pool))
            pp = pp.group_ballots()
            pp_by_bloc[bloc] = pp

        # combine the profiles
        pp = PreferenceProfile()
        for profile in pp_by_bloc.values():
            pp += profile

        if by_bloc:
            return (pp_by_bloc, pp)

        # else return the combined profiles
        else:
            return pp

    def _BT_mcmc_shortcut(
        self,
        num_ballots,
        pref_interval,
        seed_ballot,
        zero_cands={},
        verbose=False,
        burn_in_time=0,
        chain_length=None,
        BURN_IN_TIME=100000,
    ):
        """
        Sample from BT using MCMC on the shortcut ballot graph

        num_ballots (int): the number of ballots to sample
        pref_interval (dict): the preference interval to determine BT distribution
        sub_sample_length (int): how many attempts at swaps to make before saving ballot
        seed_ballot: Ballot, the seed ballot for the Markov chain
        burn_in_time (int): the number of ballots discarded in the beginning of the chain
        chain_length (int): the length of the Markov Chain. Defaults to continuous, which is num_ballots
        """
        # NOTE: Most of this has been copied from `_BT_mcmc`
        # TODO: Abstract the overlapping steps into another helper
        # function, and just pass the indices / transition probability
        # function

        if chain_length is None:
            chain_length = num_ballots

        # check that seed ballot has no ties
        for s in seed_ballot.ranking:
            if len(s) > 1:
                raise ValueError("Seed ballot contains ties")

        ballots = [-1] * num_ballots
        accept = 0
        current_ranking = list(seed_ballot.ranking)
        num_candidates = len(current_ranking)

        if verbose:
            print("MCMC on shortcut")

        burn_in_time = burn_in_time
        if verbose:
            print(f"Burn in time: {burn_in_time}")

        # precompute all the swap indices
        swap_indices = [
            tuple(sorted(random.sample(range(num_candidates), 2)))
            for _ in range(num_ballots + burn_in_time)
        ]

        for i in range(burn_in_time):
            # choose adjacent pair to propose a swap
            j1, j2 = swap_indices[i]
            j1_rank = j1 + 1
            j2_rank = j2 + 1
            if j2_rank <= j1_rank:
                raise Exception("MCMC on Shortcut: invalid ranks found")

            acceptance_prob = min(
                1,
                (pref_interval[next(iter(current_ranking[j2]))] ** (j2_rank - j1_rank))
                / (
                    pref_interval[next(iter(current_ranking[j1]))]
                    ** (j2_rank - j1_rank)
                ),
            )

            # if you accept, make the swap
            if random.random() < acceptance_prob:
                current_ranking[j1], current_ranking[j2] = (
                    current_ranking[j2],
                    current_ranking[j1],
                )
                accept += 1

        # generate MCMC sample
        for i in range(num_ballots):
            # choose adjacent pair to propose a swap
            j1, j2 = swap_indices[i]
            j1_rank = j1 + 1
            j2_rank = j2 + 1
            if j2_rank <= j1_rank:
                raise Exception("MCMC on Shortcut: invalid ranks found")

            acceptance_prob = min(
                1,
                (pref_interval[next(iter(current_ranking[j2]))] ** (j2_rank - j1_rank))
                / pref_interval[next(iter(current_ranking[j1]))] ** (j2_rank - j1_rank),
            )

            # if you accept, make the swap
            if random.random() < acceptance_prob:
                current_ranking[j1], current_ranking[j2] = (
                    current_ranking[j2],
                    current_ranking[j1],
                )
                accept += 1

            if len(zero_cands) > 0:
                ballots[i] = Ballot(ranking=current_ranking + [zero_cands])
            else:
                ballots[i] = Ballot(ranking=current_ranking)

        if verbose:
            print(
                f"Acceptance ratio as number accepted / total steps: {accept/(num_ballots+BURN_IN_TIME):.2}"
            )

        if -1 in ballots:
            raise ValueError("Some element of ballots list is not a ballot.")

        if num_ballots > chain_length:
            raise ValueError(
                "The Markov Chain length cannot be less than the number of ballots."
            )

        if verbose:
            print(f"The number of ballots before is {len(ballots)}")

        # Subsample evenly ballots
        ballots = [
            ballots[i * chain_length // num_ballots + chain_length // (2 * num_ballots)]
            for i in range(num_ballots)
        ]

        if verbose:
            print(f"The number of ballots after is {len(ballots)}")

        pp = PreferenceProfile(ballots=ballots)
        pp = pp.group_ballots()
        return pp

    def _BT_mcmc(
        self,
        num_ballots,
        pref_interval,
        seed_ballot,
        zero_cands={},
        verbose=False,
        burn_in_time=0,
        chain_length=None,
    ):
        """
        Sample from BT distribution for a given preference interval using MCMC. Defaults
        to continuous sampling and no burn-in time.

        num_ballots (int): the number of ballots to sample
        pref_interval (dict): the preference interval to determine BT distribution
        sub_sample_length (int): how many attempts at swaps to make before saving ballot
        seed_ballot: Ballot, the seed ballot for the Markov chain
        verbose: bool, if True, print the acceptance ratio of the chain
        burn_in_time (int): the number of ballots discarded in the beginning of the chain
        chain_length (int): the length of the Markov Chain. Defaults to continuous, which is num_ballots
        """
        if chain_length is None:
            chain_length = num_ballots

        # check that seed ballot has no ties
        for s in seed_ballot.ranking:
            if len(s) > 1:
                raise ValueError("Seed ballot contains ties")

        ballots = [-1] * num_ballots
        accept = 0
        current_ranking = list(seed_ballot.ranking)
        num_candidates = len(current_ranking)

        # presample swap indices
        burn_in_time = burn_in_time  # int(10e5)
        if verbose:
            print(f"Burn in time: {burn_in_time}")
        swap_indices = [
            (j1, j1 + 1)
            for j1 in random.choices(
                range(num_candidates - 1), k=num_ballots + burn_in_time
            )
        ]

        for i in range(burn_in_time):
            # choose adjacent pair to propose a swap
            j1, j2 = swap_indices[i]
            acceptance_prob = min(
                1,
                pref_interval[next(iter(current_ranking[j2]))]
                / pref_interval[next(iter(current_ranking[j1]))],
            )

            # if you accept, make the swap
            if random.random() < acceptance_prob:
                current_ranking[j1], current_ranking[j2] = (
                    current_ranking[j2],
                    current_ranking[j1],
                )
                accept += 1

        # generate MCMC sample
        for i in range(num_ballots):
            # choose adjacent pair to propose a swap
            j1, j2 = swap_indices[i]
            acceptance_prob = min(
                1,
                pref_interval[next(iter(current_ranking[j2]))]
                / pref_interval[next(iter(current_ranking[j1]))],
            )

            # if you accept, make the swap
            if random.random() < acceptance_prob:
                current_ranking[j1], current_ranking[j2] = (
                    current_ranking[j2],
                    current_ranking[j1],
                )
                accept += 1

            if len(zero_cands) > 0:
                ballots[i] = Ballot(ranking=current_ranking + [zero_cands])
            else:
                ballots[i] = Ballot(ranking=current_ranking)

        if verbose:
            print(
                f"Acceptance ratio as number accepted / total steps: {accept/(num_ballots+burn_in_time):.2}"
            )

        if -1 in ballots:
            raise ValueError("Some element of ballots list is not a ballot.")

        if num_ballots > chain_length:
            raise ValueError(
                "The Markov Chain length cannot be less than the number of ballots."
            )

        if verbose:
            print(f"The number of ballots before is {len(ballots)}")

        # Subsample evenly ballots
        ballots = [
            ballots[i * chain_length // num_ballots + chain_length // (2 * num_ballots)]
            for i in range(num_ballots)
        ]

        if verbose:
            print(f"The number of ballots after is {len(ballots)}")

        pp = PreferenceProfile(ballots=ballots)
        pp = pp.group_ballots()
        return pp

    def generate_profile_MCMC(
        self, number_of_ballots: int, verbose=False, by_bloc: bool = False
    ) -> Union[PreferenceProfile, Tuple]:
        """
        Sample from the BT distribution using Markov Chain Monte Carlo. `number_of_ballots` should
        be sufficiently large to allow for convergence of the chain.

        Args:
            number_of_ballots (int): The number of ballots to generate.
            verbose (bool, optional): If True, print the acceptance ratio of the chain. Default
                                        is False.
            by_bloc (bool): True if you want the generated profiles returned as a tuple
                ``(pp_by_bloc, pp)``, where ``pp_by_bloc`` is a dictionary with keys = bloc strings
                and values = ``PreferenceProfile`` and ``pp`` is the aggregated profile. False if
                you only want the aggregated profile. Defaults to False.

        Returns:
            Union[PreferenceProfile, Tuple]
        """

        # the number of ballots per bloc is determined by Huntington-Hill apportionment
        bloc_props = list(self.bloc_voter_prop.values())
        ballots_per_block = dict(
            zip(
                self.blocs,
                apportion.compute("huntington", bloc_props, number_of_ballots),
            )
        )

        pp_by_bloc = {b: PreferenceProfile() for b in self.blocs}

        for bloc in self.blocs:
            num_ballots = ballots_per_block[bloc]
            pref_interval = self.pref_interval_by_bloc[bloc]
            pref_interval_dict = pref_interval.interval
            non_zero_cands = pref_interval.non_zero_cands
            zero_cands = pref_interval.zero_cands

            seed_ballot = Ballot(
                ranking=tuple([frozenset({c}) for c in non_zero_cands])
            )
            pp = self._BT_mcmc(
                num_ballots,
                pref_interval_dict,
                seed_ballot,
                zero_cands=zero_cands,
                verbose=verbose,
            )

            pp_by_bloc[bloc] = pp

        # combine the profiles
        pp = PreferenceProfile()
        for profile in pp_by_bloc.values():
            pp += profile

        if by_bloc:
            return (pp_by_bloc, pp)

        # else return the combined profiles
        else:
            return pp
