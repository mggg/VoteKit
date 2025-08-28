import pytest
from votekit import Ballot
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


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
