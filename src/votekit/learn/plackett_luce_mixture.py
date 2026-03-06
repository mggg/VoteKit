from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from votekit.pref_profile import RankProfile

from typing import Optional, Any


class PlackettLuceMixture:
    """
    Name-Plackett-Luce Mixture Model fitted via expectation maximization.

    This class fits a Plackett-Luce mixture model using an Expectation Maximization (EM)
    technique, as described in
        Gormley & Murphy (2008), "Exploring Voting Blocs Within the Irish Electorate:
        A Mixture Modeling Approach," Section 4.2.

    Each EM iteration is performed using the MM (minorize-maximize) algorithm of
        Hunter (2004), "MM Algorithms for Generalized Bradley-Terry Models"

    Args:
        n_components (int, optional): Number of mixture components. Defaults to 2.
        max_iter (int, optional): Maximum number of EM iterations. Defaults to 500.
        tol (float, optional): Convergence tolerance on log-likelihood change.
            Defaults to 1e-6.
        random_state (Optional[int], optional): Seed for reproducibility. Defaults to None.

    Attributes:
        support_params_ (dict[str, NDArray[np.floating]]): Maps candidate name to an array
            of shape ``(n_components,)`` giving that candidate's support parameter in each
            component.
        mixing_weights_ (NDArray[np.floating]): Array of shape ``(n_components,)`` with the
            mixture proportions (sums to 1).
        log_likelihood_ (float): Final observed-data log-likelihood.
        converged_ (bool): Whether EM reached the tolerance before ``max_iter``.
        num_iterations_ (int): Number of EM iterations actually run.
        responsibilities_ (NDArray[np.floating]): Array of shape
            ``(n_ballots, n_components)`` with the posterior probability that each ballot
            belongs to each component.
        candidate_names_ (tuple[str, ...]): Ordered candidate names matching column indices
            in the support parameter array.
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)

        # Result attributes (populated by fit())
        self.support_params_: Optional[dict[str, NDArray[np.floating]]] = None
        self.mixing_weights_: Optional[NDArray[np.floating]] = None
        self.log_likelihood_: Optional[float] = None
        self.converged_: Optional[bool] = None
        self.num_iterations_: Optional[int] = None
        self.responsibilities_: Optional[NDArray[np.floating]] = None
        self.candidate_names_: Optional[tuple[str, ...]] = None
        self._support_params_array_: Optional[NDArray[np.floating]] = None

        # Populated by _preprocess_profile
        self._weights: Optional[NDArray[np.floating]] = None
        self._candidates: Optional[tuple[str, ...]] = None
        self._cand_to_idx: Optional[dict[str, int]] = None
        self._n_candidates: int = 0
        self._padded_rankings: Optional[NDArray[np.intp]] = None
        self._ranking_lengths: Optional[NDArray[np.intp]] = None

    def fit(self, profile: RankProfile) -> PlackettLuceMixture:
        """
        Fit the mixture model to a VoteKit ``RankProfile``.

        The fitting process cannot handle ties in ballots, so ties are broken alphabetically.

        Args:
            profile (RankProfile): VoteKit rank profile containing ranked
                ballots.

        Returns:
            PlackettLuceMixture: The fitted model.
        """

        # Convert VoteKit ballots to internal format
        (
            self._padded_rankings,
            self._ranking_lengths,
            self._weights,
            self._candidates,
            self._cand_to_idx,
        ) = self._preprocess_profile(profile)
        self._n_candidates = len(self._candidates)
        self._n_ballots = self._padded_rankings.shape[0]

        # Fit the parameters
        # TODO: Allow mutliple initializations
        (
            support_params,
            mixing_weights,
            responsibilities,
            log_likelihood,
            num_iterations,
            converged,
        ) = self._fit_single_init()

        # Record the discovered parameters
        self.support_params_ = {
            cand: support_params[:, idx] for cand, idx in self._cand_to_idx.items()
        }
        self.mixing_weights_ = mixing_weights
        self.log_likelihood_ = log_likelihood
        self.converged_ = converged
        self.num_iterations_ = num_iterations
        self.responsibilities_ = responsibilities
        self.candidate_names_ = self._candidates
        self._support_params_array_ = support_params

        return self

    def _preprocess_profile(self, profile: RankProfile) -> tuple[
        NDArray[np.intp],
        NDArray[np.intp],
        NDArray[np.floating],
        tuple[str, ...],
        dict[str, int],
    ]:
        """
        Convert a ``RankProfile`` into internal array representations.

        Each ballot becomes a row of integer candidate indices (in ranked
        order) in a padded matrix. Ties within a ranking position are broken
        alphabetically.

        Args:
            profile (RankProfile): VoteKit rank profile containing ranked ballots.

        Returns:
            tuple: A 5-tuple of ``(padded_rankings, ranking_lengths, weights,
                candidates, cand_to_idx)`` where ``padded_rankings`` has shape
                ``(n_ballots, max_ranking_length)`` padded with -1,
                ``ranking_lengths`` has shape ``(n_ballots,)``, ``weights`` is a
                1-D array of ballot weights, ``candidates`` is the ordered
                candidate names, and ``cand_to_idx`` maps each name to its
                column index.
        """
        candidates = tuple(profile.candidates)
        cand_to_idx = {c: i for i, c in enumerate(candidates)}

        # Create list-of-lists representation of rankings
        rankings: list[list[int]] = []
        weights: list[float] = []
        for ballot in profile.ballots:
            ranking_indices: list[int] = []
            if ballot.ranking is not None:
                for ranking_set in ballot.ranking:
                    # Break ties alphabetically
                    for cand in sorted(ranking_set, key=str):
                        if cand in cand_to_idx:
                            ranking_indices.append(cand_to_idx[cand])
            if ranking_indices:
                rankings.append(ranking_indices)
                weights.append(ballot.weight)

        # Create padded matrix representation
        lengths = np.array([len(r) for r in rankings], dtype=np.intp)
        max_length = int(lengths.max()) if len(lengths) > 0 else 0
        n_ballots = len(rankings)
        padded = np.full((n_ballots, max_length), -1, dtype=np.intp)
        for i, ranking in enumerate(rankings):
            padded[i, : lengths[i]] = ranking

        return padded, lengths, np.array(weights, dtype=float), candidates, cand_to_idx

    def _fit_single_init(
        self,
    ) -> tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        float,
        int,
        bool,
    ]:
        """
        Run the full EM algorithm.

        After a random initialization, the iteration order is as follows:
        E-step, check convergence, M-step, repeat.
        Convergence is declared when the absolute change in log-likelihood
        is less than ``tol``. A final E-step is always run so the returned
        responsibilities match the final parameters.

        Returns:
            tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], float, int, bool]:
                A 6-tuple of ``(support_params, mixing_weights,
                responsibilities, log_likelihood, num_iterations, converged)``.
        """
        support_params, mixing_weights = self._initialize_parameters()

        prev_log_likelihood = -np.inf
        converged = False

        for iteration in range(self.max_iter):
            # E step
            responsibilities, log_likelihood = self._e_step(
                support_params, mixing_weights
            )

            # Convergence check
            if iteration > 0 and abs(log_likelihood - prev_log_likelihood) < self.tol:
                converged = True
                break
            prev_log_likelihood = log_likelihood

            # M-step
            support_params, mixing_weights = self._m_step(
                responsibilities, support_params
            )

        # Final E-step so responsibilities match the final parameters
        responsibilities, log_likelihood = self._e_step(support_params, mixing_weights)

        return (
            support_params,
            mixing_weights,
            responsibilities,
            log_likelihood,
            iteration + 1,
            converged,
        )

    def _initialize_parameters(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Random initialisation from symmetric Dirichlet distributions.

        Each component's support vector is drawn independently from
        ``Dir(1, 1, ..., 1)`` (uniform over the simplex). Mixing weights
        start uniform.

        Returns:
            tuple[NDArray[np.floating], NDArray[np.floating]]: A pair
                ``(support_params, mixing_weights)`` with shapes
                ``(n_components, n_candidates)`` and ``(n_components,)``.
        """
        K = self.n_components
        N = self._n_candidates

        mixing_weights = np.ones(K) / K

        support_params = np.zeros((K, N))
        for k in range(K):
            support_params[k] = self._rng.dirichlet(np.ones(N))

        return support_params, mixing_weights

    def _e_step(
        self,
        support_params: NDArray[np.floating],
        mixing_weights: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], float]:
        r"""
        E-step: compute posterior responsibilities via Bayes' rule.

        The responsibility of ballot ``i`` by component ``k`` is computed by:
        ``\tau_k * f(x_i | p_k) / \sum_{k'} \tau_{k'} * f(x_i | p_{k'})``
        where ``\tau`` is the vector of mixture components, and ``p_k`` is the
        estimated utility vector for bloc ``k``.

        Args:
            support_params (NDArray[np.floating]): Support parameter matrix,
                shape ``(n_components, n_candidates)``.
            mixing_weights (NDArray[np.floating]): Mixture proportions,
                shape ``(n_components,)``.

        Returns:
            tuple[NDArray[np.floating], float]: A pair
                ``(responsibilities, log_likelihood)`` where ``responsibilities``
                has shape ``(n_ballots, n_components)`` and ``log_likelihood`` is
                the weighted observed-data log-likelihood.
        """
        # log f(x_i | p_k) for all (i, k)
        log_liks = self._compute_log_likelihoods(
            support_params
        )  # (n_ballots, n_components)

        # log(pi_k * f(x_i | p_k)) = log pi_k + log f(x_i | p_k)
        log_weights = np.log(mixing_weights + 1e-300)  # (n_components,)
        log_joint = log_liks + log_weights[np.newaxis, :]  # (n_ballots, n_components)

        # log sum_k pi_k f(x_i | p_k) -- the marginal likelihood per ballot
        log_marginal = logsumexp(log_joint, axis=1)  # (n_ballots,)

        # Responsibilities: exp(log_joint_ik - log_marginal_i)
        log_resp = log_joint - log_marginal[:, np.newaxis]
        responsibilities = np.exp(log_resp)

        # Observed-data log-likelihood, weighted by ballot weights
        log_likelihood = np.sum(self._weights * log_marginal)

        return responsibilities, log_likelihood

    def _compute_log_likelihoods(
        self, support_params: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Build the ``(n_ballots, n_components)`` matrix of log-likelihoods.

        Entry ``[i, k]`` is ``log P(ranking_i | support_k)``.

        Args:
            support_params (NDArray[np.floating]): Support parameter matrix,
                shape ``(n_components, n_candidates)``.

        Returns:
            NDArray[np.floating]: Log-likelihood matrix, shape
                ``(n_ballots, n_components)``.
        """
        assert self._padded_rankings is not None
        assert self._ranking_lengths is not None
        padded_rankings = self._padded_rankings
        lengths = self._ranking_lengths
        L_max = padded_rankings.shape[1]

        # Tricky vectorized way to compute all log-likelihoods at once
        log_liks = np.zeros((self._n_ballots, self.n_components))
        for k in range(self.n_components):
            p = support_params[k]
            log_p = np.log(np.maximum(p, 1e-300))  # log-supports for this component

            # Cumulative remaining support: starts at sum(p)
            remaining = np.full(self._n_ballots, p.sum())

            for pos in range(L_max):
                # Find indices of ballots of length at least pos
                valid = pos < lengths
                if not valid.any():
                    break
                valid_idx = np.where(valid)[0]

                # The candidate chosen at this position for each valid ballot
                chosen_candidate = padded_rankings[valid_idx, pos]
                chosen_support = p[chosen_candidate]
                remaining_support = remaining[valid_idx]

                # PL probability at position j is p(chosen) / sum(p remaining).
                # In log-space: log p(chosen) - log(remaining support).
                # Guard against zero support or zero remaining (impossible
                # ranking under this component) by sending those to -inf.
                computable = (chosen_support > 0) & (remaining_support > 0)
                log_liks[valid_idx[computable], k] += log_p[
                    chosen_candidate[computable]
                ] - np.log(remaining_support[computable])
                log_liks[valid_idx[~computable], k] = -np.inf

                # Remove the chosen candidate's support from the running total
                remaining[valid_idx] -= chosen_support

        return log_liks

    def _m_step(
        self,
        responsibilities: NDArray[np.floating],
        support_params: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        M-step: re-estimate all parameters given responsibilities.

        1. Mixing weights are updated as the weighted average of
           responsibilities.
        2. Support parameters for each component are updated via the
           MM algorithm (see ``_mm_update_support``).

        Args:
            responsibilities (NDArray[np.floating]): Responsibility matrix,
                shape ``(n_ballots, n_components)``.
            support_params (NDArray[np.floating]): Support parameter matrix,
                shape ``(n_components, n_candidates)``.

        Returns:
            tuple[NDArray[np.floating], NDArray[np.floating]]: A pair
                ``(new_support, new_mixing)`` with shapes
                ``(n_components, n_candidates)`` and ``(n_components,)``.
        """
        K = self.n_components
        assert self._weights is not None

        # Update mixing weights
        total_weight = self._weights.sum()
        new_mixing = np.array(
            [
                np.sum(self._weights * responsibilities[:, k]) / total_weight
                for k in range(K)
            ]
        )
        # new_mixing should theoretically sum to 1. We'll enforce that for numerical safety.
        new_mixing = new_mixing / new_mixing.sum()

        # Update support parameters per component
        new_support = support_params.copy()
        for k in range(K):
            new_support[k] = self._mm_update_support(
                self._weights,
                responsibilities[:, k],
                support_params[k],
            )

        return new_support, new_mixing

    def _mm_update_support(
        self,
        weights: NDArray[np.floating],
        z_k: NDArray[np.floating],
        current_p: NDArray[np.floating],
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> NDArray[np.floating]:
        """
        Update one component's support parameters using the MM algorithm.

        Hunter (2004) derives a minorisation of the PL log-likelihood that
        yields a simple closed-form update for each candidate ``j``:
        ``p_j_new = numerator_j / denominator_j``.

        Args:
            weights (NDArray[np.floating]): Ballot weights, shape ``(n_ballots,)``.
            z_k (NDArray[np.floating]): Responsibilities for this component
                (from the E-step), shape ``(n_ballots,)``.
            current_p (NDArray[np.floating]): Current support parameters,
                shape ``(n_candidates,)``.
            max_iter (int, optional): Max inner MM iterations. Defaults to 100.
            tol (float, optional): Convergence tolerance on max absolute change.
                Defaults to 1e-8.

        Returns:
            NDArray[np.floating]: Updated support parameters, shape
                ``(n_candidates,)``, summing to 1.
        """
        assert self._padded_rankings is not None
        assert self._ranking_lengths is not None
        n_candidates = len(current_p)
        n_ballots = len(weights)
        padded_rankings = self._padded_rankings  # (n_ballots, max_rank_len)
        ranking_lengths = self._ranking_lengths  # (n_ballots,)
        max_rank_len = padded_rankings.shape[1]

        # Effective weight per ballot: ballot weight * responsibility for
        # this component.  Ballots with negligible effective weight are
        # excluded from all inner loops.
        effective_weight = weights * z_k  # (n_ballots,)
        active_ballot = effective_weight > 1e-300

        support = current_p.copy()

        # Numerator (fixed across MM iterations)
        numerator = np.zeros(n_candidates)
        for pos in range(max_rank_len):
            ballots_at_pos = active_ballot & (pos < ranking_lengths)
            if not ballots_at_pos.any():
                break
            ballot_idx = np.where(ballots_at_pos)[0]
            np.add.at(
                numerator,
                padded_rankings[ballot_idx, pos],
                effective_weight[ballot_idx],
            )

        # Cache which ballots are active at each ranking position (the
        # ranking structure doesn't change between MM iterations).
        active_ballots_by_pos: list[NDArray[np.intp]] = []
        for pos in range(max_rank_len):
            ballots_at_pos = active_ballot & (pos < ranking_lengths)
            ballot_idx = np.where(ballots_at_pos)[0]
            if len(ballot_idx) == 0:
                break
            active_ballots_by_pos.append(ballot_idx)

        # MM iteration loop
        for _ in range(max_iter):
            denominator = np.zeros(n_candidates)

            # Track the sum of support over *remaining* (not-yet-ranked)
            # candidates for each ballot.  Starts at sum(support) because
            # no candidates have been "removed" yet.
            remaining_support = np.full(n_ballots, support.sum())

            for pos in range(len(active_ballots_by_pos)):
                ballot_idx = active_ballots_by_pos[pos]
                chosen_candidate = padded_rankings[ballot_idx, pos]

                # Each ballot contributes  w_i / remaining_support_i  to the
                # denominator of every candidate still in contention.
                ballot_remaining = remaining_support[ballot_idx]
                position_contrib = np.zeros(len(ballot_idx))
                computable = ballot_remaining > 0
                position_contrib[computable] = (
                    effective_weight[ballot_idx[computable]]
                    / ballot_remaining[computable]
                )

                # Broadcast: every candidate gets the total contribution,
                # then we subtract it back out for candidates already ranked
                # at earlier positions (they are no longer "remaining").
                denominator += position_contrib.sum()
                for earlier_pos in range(pos):
                    already_ranked = padded_rankings[ballot_idx, earlier_pos]
                    np.subtract.at(denominator, already_ranked, position_contrib)

                # Remove the chosen candidate's support from the running total
                remaining_support[ballot_idx] -= support[chosen_candidate]

            # Apply the MM update: p_j_new = numerator_j / denominator_j
            updated_support = np.full(n_candidates, 1.0 / n_candidates)
            has_denom = denominator > 0
            updated_support[has_denom] = numerator[has_denom] / denominator[has_denom]
            updated_support = updated_support / updated_support.sum()

            if np.max(np.abs(updated_support - support)) < tol:
                break
            support = updated_support

        return support

    def _log_pl_probability(
        self, ranking: list[int], support: NDArray[np.floating]
    ) -> float:
        """
        Log-probability of a single ranking under one PL component.

        Implements Equation (1) of Gormley & Murphy (2008):
        ``P(x | p) = prod_{t=1}^{n} p_{c_t} / sum_{s=t}^{N} p_{c_s}``
        where ``c_t`` is the candidate ranked at position *t* and the
        denominator sums over all candidates not yet ranked.

        All arithmetic is done in log-space to avoid numerical underflow
        when the number of candidates is large.

        Args:
            ranking (list[int]): Candidate indices in ranked order
                (first = most preferred).
            support (NDArray[np.floating]): Support parameters for this
                component, shape ``(n_candidates,)``, summing to 1.

        Returns:
            float: ``log P(ranking | support)``.
        """
        if len(ranking) == 0:
            return 0.0

        log_prob = 0.0
        N = len(support)

        # Boolean mask tracking which candidates are still available
        available = np.ones(N, dtype=bool)

        for idx in ranking:
            remaining_support = support[available].sum()
            if remaining_support <= 0 or support[idx] <= 0:
                return -np.inf  # impossible ranking under this component

            # log(p_idx / sum remaining) = log(p_idx) - log(sum remaining)
            log_prob += np.log(support[idx]) - np.log(remaining_support)
            available[idx] = False

        return log_prob

    def name_pl_parameters(self) -> dict[str, Any]:
        """
        Format parameters for VoteKit's ``BlocSlateConfig``.

        All candidates are placed in a single slate called ``"All"``.

        Returns:
            dict[str, Any]: Dictionary ready to unpack into
                ``BlocSlateConfig(n_voters=..., **result)``.

        Example::

            config = BlocSlateConfig(n_voters=100,
                **model.name_pl_parameters())
            profile = name_pl_profile_generator(config)
        """
        assert self.mixing_weights_ is not None
        assert self.candidate_names_ is not None
        assert self.support_params_ is not None
        K = len(self.mixing_weights_)
        candidates = list(self.candidate_names_)
        bloc_names = [f"bloc_{k}" for k in range(K)]

        # With a single slate, cohesion is always 1.0 and preferences
        # are the raw support parameters.
        return {
            "slate_to_candidates": {"All": candidates},
            "bloc_proportions": {
                name: float(self.mixing_weights_[k])
                for k, name in enumerate(bloc_names)
            },
            "cohesion_mapping": {name: {"All": 1.0} for name in bloc_names},
            "preference_mapping": {
                name: {
                    "All": {c: float(self.support_params_[c][k]) for c in candidates}
                }
                for k, name in enumerate(bloc_names)
            },
        }
