from votekit.pref_interval import PreferenceInterval, combine_preference_intervals
import pytest


def test_from_dirichlet():
    # check that we don't get zero support cands randomly
    for _ in range(10_000):
        for alpha in [0.001, 0.01, 1, 10, 100]:
            pi = PreferenceInterval.from_dirichlet(
                candidates=["A", "B", "C"], alpha=alpha
            )
            assert isinstance(pi, PreferenceInterval)


def test_from_interval():
    pi = PreferenceInterval(interval={"A": 0.2, "B": 0.3, "C": 0.5})
    assert isinstance(pi, PreferenceInterval)


def test_normalize():
    interval_1 = PreferenceInterval(interval={"A": 2, "B": 3, "C": 5})
    interval_2 = PreferenceInterval(interval={"A": 0.2, "B": 0.3, "C": 0.5})

    assert interval_1 == interval_2


def test_zero_support_error():
    with pytest.raises(ValueError, match="Candidate A has zero support."):
        PreferenceInterval(interval={"A": 0, "B": 1})


def test_eq():
    interval_1 = PreferenceInterval(interval={"A": 2, "B": 3, "C": 5})
    interval_2 = PreferenceInterval(interval={"A": 0.2, "B": 0.3, "C": 0.5})
    interval_3 = PreferenceInterval(interval={"A": 2, "B": 3, "C": 6})

    # same after normalizing
    assert interval_1 == interval_2

    # not same interval
    assert interval_1 != interval_3


def test_combine_error():
    interval_1 = PreferenceInterval(interval={"A": 2, "B": 3, "C": 6})
    interval_2 = PreferenceInterval(interval={"A": 0.2, "B": 0.3, "C": 0.5})
    interval_3 = PreferenceInterval(interval={"E": 2, "F": 4})

    with pytest.raises(ValueError, match="Intervals must have disjoint candidate sets"):
        combine_preference_intervals([interval_1, interval_2], [1 / 2, 1 / 2])

    with pytest.raises(ValueError, match="Proportions must sum to 1."):
        combine_preference_intervals([interval_1, interval_3], [2 / 3, 4 / 3])


def test_combine():
    interval_1 = PreferenceInterval(interval={"A": 2, "B": 3, "C": 6})
    interval_2 = PreferenceInterval(interval={"D": 1})
    interval_3 = PreferenceInterval(interval={"E": 2, "F": 4})
    combined_pi = combine_preference_intervals(
        [interval_1, interval_2, interval_3], [1 / 3, 1 / 3, 1 / 3]
    )
    true_result = PreferenceInterval(
        interval={
            "A": 2 / 33,
            "B": 3 / 33,
            "C": 6 / 33,
            "D": 1 / 3,
            "E": 1 / 9,
            "F": 2 / 9,
        }
    )

    assert combined_pi == true_result
