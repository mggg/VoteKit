import numpy as np
from typing import Optional, Union, Tuple, Callable, Dict, Any

from votekit.pref_profile import PreferenceProfile
from votekit.metrics import euclidean_dist
from votekit.ballot_generator import BallotGenerator


class OneDimSpatial(BallotGenerator):
    """
    1-D spatial model for ballot generation. Assumes the candidates are normally distributed on
    the real line. Then voters are also normally distributed, and vote based on Euclidean distance
    to the candidates.

    Args:
        candidates (list): List of candidate strings.

    Attributes:
        candidates (list): List of candidate strings.

    """

    def generate_profile(
        self, number_of_ballots: int, by_bloc: bool = False
    ) -> Union[PreferenceProfile, Tuple]:
        """
        Args:
            number_of_ballots (int): The number of ballots to generate.

        Returns:
            Union[PreferenceProfile, Tuple]
        """
        candidate_position_dict = {c: np.random.normal(0, 1) for c in self.candidates}
        voter_positions = np.random.normal(0, 1, number_of_ballots)

        ballot_pool = []

        for vp in voter_positions:
            distance_dict = {
                c: abs(v - vp) for c, v, in candidate_position_dict.items()
            }
            candidate_order = sorted(
                distance_dict, key=lambda x: float(distance_dict.__getitem__(x))
            )
            ballot_pool.append(candidate_order)

        return self.ballot_pool_to_profile(ballot_pool, self.candidates)


class Spatial(BallotGenerator):
    """
    Spatial model for ballot generation. In some metric space determined
    by an input distance function, randomly sample each voter's and
    each candidate's positions from input voter and candidate distributions.
    Using generate_profile() outputs a ranked profile which is consistent
    with the sampled positions (respects distances).

    Args:
        candidates (list[str]): List of candidate strings.
        voter_dist (Callable[..., np.ndarray], optional): Distribution to sample a single
            voter's position from, defaults to uniform distribution.
        voter_dist_kwargs: (Optional[Dict[str, Any]], optional): Keyword args to be passed to
            voter_dist, defaults to None, which creates the unif(0,1) distribution in 2 dimensions.
        candidate_dist: (Callable[..., np.ndarray], optional): Distribution to sample a
            single candidate's position from, defaults to uniform distribution.
        candidate_dist_kwargs: (Optional[Dict[str, Any]], optional): Keyword args to be passed
            to candidate_dist, defaults to None, which creates the unif(0,1)
            distribution in 2 dimensions.
        distance: (Callable[[np.ndarray, np.ndarray], float]], optional):
            Computes distance between a voter and a candidate,
            defaults to euclidean distance.
    Attributes:
        candidates (list[str]): List of candidate strings.
        voter_dist (Callable[..., np.ndarray], optional): Distribution to sample a single
            voter's position from, defaults to uniform distribution.
        voter_dist_kwargs: (Optional[Dict[str, Any]], optional): Keyword args to be passed to
            voter_dist, defaults to None, which creates the unif(0,1) distribution in 2 dimensions.
        candidate_dist: (Callable[..., np.ndarray], optional): Distribution to sample a
            single candidate's position from, defaults to uniform distribution.
        candidate_dist_kwargs: (Optional[Dict[str, Any]], optional): Keyword args to be passed
            to candidate_dist, defaults to None, which creates the unif(0,1)
            distribution in 2 dimensions.
        distance: (Callable[[np.ndarray, np.ndarray], float]], optional):
            Computes distance between a voter and a candidate,
            defaults to euclidean distance.
    """

    def __init__(
        self,
        candidates: list[str],
        voter_dist: Callable[..., np.ndarray] = np.random.uniform,
        voter_dist_kwargs: Optional[Dict[str, Any]] = None,
        candidate_dist: Callable[..., np.ndarray] = np.random.uniform,
        candidate_dist_kwargs: Optional[Dict[str, Any]] = None,
        distance: Callable[[np.ndarray, np.ndarray], float] = euclidean_dist,
    ):
        super().__init__(candidates=candidates)
        self.voter_dist = voter_dist
        self.candidate_dist = candidate_dist

        if voter_dist_kwargs is None:
            if voter_dist is np.random.uniform:
                voter_dist_kwargs = {"low": 0.0, "high": 1.0, "size": 2.0}
            else:
                voter_dist_kwargs = {}

        try:
            self.voter_dist(**voter_dist_kwargs)
        except TypeError:
            raise TypeError("Invalid kwargs for the voter distribution.")

        self.voter_dist_kwargs = voter_dist_kwargs

        if candidate_dist_kwargs is None:
            if candidate_dist is np.random.uniform:
                candidate_dist_kwargs = {"low": 0.0, "high": 1.0, "size": 2.0}
            else:
                candidate_dist_kwargs = {}

        try:
            self.candidate_dist(**candidate_dist_kwargs)
        except TypeError:
            raise TypeError("Invalid kwargs for the candidate distribution.")

        self.candidate_dist_kwargs = candidate_dist_kwargs

        try:
            v = self.voter_dist(**self.voter_dist_kwargs)
            c = self.candidate_dist(**self.candidate_dist_kwargs)
            distance(v, c)
        except TypeError:
            raise TypeError(
                "Distance function is invalid or incompatible "
                "with voter/candidate distributions."
            )

        self.distance = distance

    def generate_profile(
        self, number_of_ballots: int, by_bloc: bool = False
    ) -> Tuple[PreferenceProfile, dict[str, np.ndarray], np.ndarray]:
        """
        Samples a metric position for number_of_ballots voters from
        the voter distribution. Samples a metric position for each candidate
        from the input candidate distribution. With sampled
        positions, this method then creates a ranked PreferenceProfile in which
        voter's preferences are consistent with their distances to the candidates
        in the metric space.

        Args:
            number_of_ballots (int): The number of ballots to generate.
            by_bloc (bool): Dummy variable from parent class.

        Returns:
            Tuple[PreferenceProfile, dict[str, numpy.ndarray], numpy.ndarray]:
                A tuple containing the preference profile object,
                a dictionary with each candidate's position in the metric
                space, and a matrix where each row is a single voter's position
                in the metric space.
        """

        candidate_position_dict = {
            c: self.candidate_dist(**self.candidate_dist_kwargs)
            for c in self.candidates
        }
        voter_positions = np.array(
            [
                self.voter_dist(**self.voter_dist_kwargs)
                for v in range(number_of_ballots)
            ]
        )

        ballot_pool = [["c"] * len(self.candidates) for _ in range(number_of_ballots)]
        for v in range(number_of_ballots):
            distance_dict = {
                c: self.distance(voter_positions[v], c_position)
                for c, c_position in candidate_position_dict.items()
            }
            candidate_order = sorted(distance_dict, key=distance_dict.__getitem__)
            ballot_pool[v] = candidate_order

        return (
            self.ballot_pool_to_profile(ballot_pool, self.candidates),
            candidate_position_dict,
            voter_positions,
        )


class ClusteredSpatial(BallotGenerator):
    """
    Clustered spatial model for ballot generation. In some metric space
    determined by an input distance function, randomly sample
    each candidate's positions from input candidate distribution. Then
    sample voters's positions from a distribution centered around each
    of the candidate's positions.

    NOTE: We currently only support the following list of voter distributions:
    [np.random.normal, np.random.laplace, np.random.logistic, np.random.gumbel],
    which is the complete list of numpy distributions that accept a 'loc' parameter allowing
    us to center the distribution around each candidate. For more
    information on numpy supported distributions and their parameters, please visit:
    https://numpy.org/doc/1.16/reference/routines.random.html.

    Args:
        candidates (list[str]): List of candidate strings.
        voter_dist (Callable[..., np.ndarray], optional): Distribution to sample a single
            voter's position from, defaults to normal(0,1) distribution.
        voter_dist_kwargs: (Optional[dict[str, Any]], optional): Keyword args to be passed to
            voter_dist, defaults to None, which creates the unif(0,1) distribution in 2 dimensions.
        candidate_dist: (Callable[..., np.ndarray], optional): Distribution to sample a
            single candidate's position from, defaults to uniform distribution.
        candidate_dist_kwargs: (Optional[Dict[str, float]], optional): Keyword args to be passed
            to candidate_dist, defaults None which creates the unif(0,1)
            distribution in 2 dimensions.
        distance: (Callable[[np.ndarray, np.ndarray], float]], optional):
            Computes distance between a voter and a candidate,
            defaults to euclidean distance.
    Attributes:
        candidates (list[str]): List of candidate strings.
        voter_dist (Callable[..., np.ndarray], optional): Distribution to sample a single
            voter's position from, defaults to uniform distribution.
        voter_dist_kwargs: (Optional[dict[str, Any]], optional): Keyword args to be passed to
            voter_dist, defaults to None, which creates the unif(0,1) distribution in 2 dimensions.
        candidate_dist: (Callable[..., np.ndarray], optional): Distribution to sample a
            single candidate's position from, defaults to uniform distribution.
        candidate_dist_kwargs: (Optional[Dict[str, float]], optional): Keyword args to be passed
            to candidate_dist, defaults None which creates the unif(0,1)
            distribution in 2 dimensions.
        distance: (Callable[[np.ndarray, np.ndarray], float]], optional):
            Computes distance between a voter and a candidate,
            defaults to euclidean distance.
    """

    def __init__(
        self,
        candidates: list[str],
        voter_dist: Callable[..., np.ndarray] = np.random.normal,
        voter_dist_kwargs: Optional[Dict[str, Any]] = None,
        candidate_dist: Callable[..., np.ndarray] = np.random.uniform,
        candidate_dist_kwargs: Optional[Dict[str, Any]] = None,
        distance: Callable[[np.ndarray, np.ndarray], float] = euclidean_dist,
    ):
        super().__init__(candidates=candidates)
        self.candidate_dist = candidate_dist
        self.voter_dist = voter_dist

        if voter_dist_kwargs is None:
            if self.voter_dist is np.random.normal:
                voter_dist_kwargs = {
                    "loc": 0,
                    "std": np.array(1.0),
                    "size": np.array(2.0),
                }
            else:
                voter_dist_kwargs = {}

        if voter_dist.__name__ not in ["normal", "laplace", "logistic", "gumbel"]:
            raise ValueError("Input voter distribution not supported.")

        try:
            voter_dist_kwargs["loc"] = 0
            self.voter_dist(**voter_dist_kwargs)
        except TypeError:
            raise TypeError("Invalid kwargs for the voter distribution.")

        self.voter_dist_kwargs = voter_dist_kwargs

        if candidate_dist_kwargs is None:
            if self.candidate_dist is np.random.uniform:
                candidate_dist_kwargs = {"low": 0.0, "high": 1.0, "size": 2.0}
            else:
                candidate_dist_kwargs = {}

        try:
            self.candidate_dist(**candidate_dist_kwargs)
        except TypeError:
            raise TypeError("Invalid kwargs for the candidate distribution.")

        self.candidate_dist_kwargs = candidate_dist_kwargs

        try:
            v = self.voter_dist(**self.voter_dist_kwargs)
            c = self.candidate_dist(**self.candidate_dist_kwargs)
            distance(v, c)
        except TypeError:
            raise TypeError(
                "Distance function is invalid or incompatible "
                "with voter/candidate distributions."
            )

        self.distance = distance

    def generate_profile_with_dict(
        self, number_of_ballots: dict[str, int], by_bloc: bool = False
    ) -> Tuple[PreferenceProfile, dict[str, np.ndarray], np.ndarray]:
        """
        Samples a metric position for each candidate
        from the input candidate distribution. For each candidate, then sample
        number_of_ballots[candidate] metric positions for voters
        which will be centered around the candidate.
        With sampled positions, this method then creates a ranked PreferenceProfile in which
        voter's preferences are consistent with their distances to the candidates
        in the metric space.

        Args:
            number_of_ballots (dict[str, int]): The number of voters attributed
                        to each candidate {candidate string: # voters}.
            by_bloc (bool): Dummy variable from parent class.

        Returns:
            Tuple[PreferenceProfile, dict[str, numpy.ndarray], numpy.ndarray]:
                A tuple containing the preference profile object,
                a dictionary with each candidate's position in the metric
                space, and a matrix where each row is a single voter's position
                in the metric space.
        """

        candidate_position_dict = {
            c: self.candidate_dist(**self.candidate_dist_kwargs)
            for c in self.candidates
        }

        n_voters = sum(number_of_ballots.values())
        voter_positions = [np.zeros(2) for _ in range(n_voters)]
        vidx = 0
        for c, c_position in candidate_position_dict.items():
            for v in range(number_of_ballots[c]):
                self.voter_dist_kwargs["loc"] = c_position
                voter_positions[vidx] = self.voter_dist(**self.voter_dist_kwargs)
                vidx += 1

        ballot_pool = [
            ["c"] * len(self.candidates) for _ in range(len(voter_positions))
        ]
        for v in range(len(voter_positions)):
            v_position = voter_positions[v]
            distance_dict = {
                c: self.distance(v_position, c_position)
                for c, c_position, in candidate_position_dict.items()
            }
            candidate_order = sorted(distance_dict, key=distance_dict.__getitem__)
            ballot_pool[v] = candidate_order

        voter_positions_array = np.vstack(voter_positions)

        return (
            self.ballot_pool_to_profile(ballot_pool, self.candidates),
            candidate_position_dict,
            voter_positions_array,
        )
