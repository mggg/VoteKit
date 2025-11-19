import pytest
from votekit import Ballot
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math
from votekit.pref_profile import RankProfile, ScoreProfile
from scipy import stats
from votekit.ballot_generator import BlocSlateConfig
from votekit.pref_interval import PreferenceInterval


# NOTE: Lock down the rendering for snapshot tests
@pytest.fixture(autouse=True)
def stable_matplotlib_rc():
    mpl.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 100,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.figsize": (4.8, 3.2),
            "savefig.bbox": "tight",
        }
    )
    sns.set_theme(style="whitegrid", rc={})
    yield
    plt.close("all")


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run tests marked slow"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="use --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def partitions_with_permutations_of_size(set_, subset_size):
    """
    Generate all partitions of subsets of a given size with all permutations of each partition.

    Args:
        set_ (set): The input set to partition and permute.
        subset_size (int): The size of the subsets for which to generate partitions.

    Returns:
        List[List[List]]: A list of all partitions with permutations of the specified subset size.
    """

    def partitions(set_):
        """Generate all partitions of a set."""
        if not set_:
            return [[]]
        result = []
        for i in range(1, len(set_) + 1):
            for combination in itertools.combinations(set_, i):
                rest = set_ - set(combination)
                for subpartition in partitions(rest):
                    result.append([frozenset(combination)] + subpartition)
        return result

    # Generate all subsets of the desired size
    subsets_of_size = list(itertools.combinations(set_, subset_size))

    all_partitions_with_permutations = []

    for subset in subsets_of_size:
        subset_set = set(subset)
        all_partitions = partitions(subset_set)
        for partition in all_partitions:
            for perm in itertools.permutations(partition):
                all_partitions_with_permutations.append(tuple(perm))

    return all_partitions_with_permutations


@pytest.fixture
def all_possible_ranked_ballots():

    def inner(cand_set):
        if len(cand_set) > 5:
            raise ValueError("Can only generate ballots for sets of size 5 or less.")

        results = []
        for i in range(1, len(cand_set) + 1):
            full_subsets = {
                x: 0 for x in partitions_with_permutations_of_size(cand_set, i)
            }

            results += list(list(x) for x in full_subsets)

        results = [Ballot(ranking=[set(x) for x in lst]) for lst in results]
        return results

    return inner


@pytest.fixture
def bloc_order_probs_slate_first():
    def _bloc_order_probs_slate_first(slate, ballot_frequencies):
        slate_first_count = sum(
            [freq for ballot, freq in ballot_frequencies.items() if ballot[0] == slate]
        )
        prob_ballot_given_slate_first = {
            ballot: freq / slate_first_count
            for ballot, freq in ballot_frequencies.items()
            if ballot[0] == slate
        }
        return prob_ballot_given_slate_first

    return _bloc_order_probs_slate_first


@pytest.fixture
def compute_pl_prob():
    def _compute_pl_prob(perm, interval):
        pref_interval = interval.copy()
        prob = 1
        for c in perm:
            if sum(pref_interval.values()) == 0:
                prob *= 1 / math.factorial(len(pref_interval))
            else:
                prob *= pref_interval[c] / sum(pref_interval.values())
            del pref_interval[c]
        return prob

    return _compute_pl_prob


def binomial_confidence_interval(probability, n_attempts, alpha=0.95):
    # Calculate the mean and standard deviation of the binomial distribution
    mean = n_attempts * probability
    std_dev = math.sqrt(n_attempts * probability * (1 - probability))

    # Calculate the confidence interval
    z_score = stats.norm.ppf((1 + alpha) / 2)  # Z-score for 99% confidence level
    margin_of_error = z_score * (std_dev)
    conf_interval = (mean - margin_of_error, mean + margin_of_error)

    return conf_interval


@pytest.fixture
def do_ballot_probs_match_ballot_dist_rank_profile():
    def _do_ballot_probs_match_ballot_dist_rank_profile(
        ballot_prob_dict: dict, generated_profile: RankProfile, alpha=0.95
    ):
        n_ballots = generated_profile.total_ballot_wt
        ballot_conf_dict = {
            b: binomial_confidence_interval(p, n_attempts=int(n_ballots), alpha=alpha)
            for b, p in ballot_prob_dict.items()
        }

        failed = 0

        for b in ballot_conf_dict.keys():
            b_list = [{c} for c in b]
            ballot = next(
                (
                    element
                    for element in generated_profile.ballots
                    if element.ranking == b_list
                ),
                None,
            )
            ballot_weight = 0.0
            if ballot is not None:
                ballot_weight = ballot.weight
            if not (
                int(ballot_conf_dict[b][0])
                <= ballot_weight
                <= int(ballot_conf_dict[b][1])
            ):
                failed += 1

        # allow for small margin of error given confidence intereval
        failure_thresold = round((1 - alpha) * n_ballots)
        return failed <= failure_thresold

    return _do_ballot_probs_match_ballot_dist_rank_profile


# FIX: This needs to be made better for score profiles
@pytest.fixture
def do_ballot_probs_match_ballot_dist_score_profile():
    def _do_ballot_probs_match_ballot_dist_score_profile(
        ballot_prob_dict: dict, generated_profile: ScoreProfile, alpha=0.95
    ):
        n_ballots = generated_profile.total_ballot_wt
        ballot_conf_dict = {
            b: binomial_confidence_interval(p, n_attempts=int(n_ballots), alpha=alpha)
            for b, p in ballot_prob_dict.items()
        }

        failed = 0

        for b in ballot_conf_dict.keys():
            ballot = next(
                (element for element in generated_profile.ballots),
                None,
            )
            ballot_weight = 0.0
            if ballot is not None:
                ballot_weight = ballot.weight
            if not (
                int(ballot_conf_dict[b][0])
                <= ballot_weight
                <= int(ballot_conf_dict[b][1])
            ):
                failed += 1

        # allow for small margin of error given confidence intereval
        failure_thresold = round((1 - alpha) * n_ballots)
        return failed <= failure_thresold

    return _do_ballot_probs_match_ballot_dist_score_profile


@pytest.fixture
def two_bloc_two_slate_config():
    return BlocSlateConfig(
        n_voters=100_000,
        slate_to_candidates={"A": ["A1", "A2"], "B": ["B1", "B2"]},
        bloc_proportions={"X": 0.7, "Y": 0.3},
        preference_mapping={
            "X": {
                "A": PreferenceInterval({"A1": 0.4, "A2": 0.3}),
                "B": PreferenceInterval({"B1": 0.2, "B2": 0.1}),
            },
            "Y": {
                "A": PreferenceInterval({"A1": 0.2, "A2": 0.2}),
                "B": PreferenceInterval({"B1": 0.3, "B2": 0.3}),
            },
        },
        cohesion_mapping={"X": {"A": 0.7, "B": 0.3}, "Y": {"B": 0.9, "A": 0.1}},
    )


@pytest.fixture
def two_bloc_two_slate_config_cambridge():
    return BlocSlateConfig(
        n_voters=100_000,
        slate_to_candidates={"X": ["X1", "X2"], "Y": ["Y1", "Y2"]},
        bloc_proportions={"X": 0.6, "Y": 0.4},
        preference_mapping={
            "X": {
                "X": PreferenceInterval({"X1": 0.4, "X2": 0.3}),
                "Y": PreferenceInterval({"Y1": 0.2, "Y2": 0.1}),
            },
            "Y": {
                "X": PreferenceInterval({"X1": 0.2, "X2": 0.2}),
                "Y": PreferenceInterval({"Y1": 0.3, "Y2": 0.3}),
            },
        },
        cohesion_mapping={"X": {"X": 0.7, "Y": 0.3}, "Y": {"Y": 0.9, "X": 0.1}},
    )


@pytest.fixture
def one_bloc_three_slate_config():
    return BlocSlateConfig(
        n_voters=100_000,
        slate_to_candidates={
            "A": ["A1", "A2", "A3"],
            "B": ["B1", "B2"],
            "C": ["C1", "C2"],
        },
        bloc_proportions={"X": 1},
        preference_mapping={
            "X": {
                "A": PreferenceInterval({"A1": 1 / 2, "A2": 1 / 2, "A3": 1 / 2}),
                "B": PreferenceInterval({"B1": 1 / 3, "B2": 2 / 3}),
                "C": PreferenceInterval({"C1": 1 / 4, "C2": 3 / 4}),
            }
        },
        cohesion_mapping={
            "X": {"A": 0.7, "B": 0.2, "C": 0.1},
        },
    )
