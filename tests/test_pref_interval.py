from votekit.pref_interval import PreferenceInterval, combine_preference_intervals
import pytest


def test_from_dirichlet():
    pi = PreferenceInterval.from_dirichlet(candidates=["A", "B", "C"], alpha=0.01)

    assert isinstance(pi, PreferenceInterval)


def test_from_interval():
    pi = PreferenceInterval(interval={"A": 0.2, "B": 0.3, "C": 0.5})
    assert isinstance(pi, PreferenceInterval)


def test_normalize():
    interval_1 = PreferenceInterval(interval={"A": 2, "B": 3, "C": 5})
    interval_2 = PreferenceInterval(interval={"A": 0.2, "B": 0.3, "C": 0.5})

    assert interval_1 == interval_2

    with pytest.raises(
        ZeroDivisionError, match="There are no candidates with non-zero support."
    ):
        PreferenceInterval(interval={"A": 0, "B": 0})


def test_eq():
    interval_1 = PreferenceInterval(interval={"A": 2, "B": 3, "C": 5})
    interval_2 = PreferenceInterval(interval={"A": 0.2, "B": 0.3, "C": 0.5})
    interval_3 = PreferenceInterval(interval={"A": 2, "B": 3, "C": 6, "D": 0})
    interval_4 = PreferenceInterval(interval={"A": 2, "B": 3, "C": 6})

    # same after normalizing
    assert interval_1 == interval_2

    # not same interval
    assert interval_1 != interval_3

    # not same non-zero cands
    assert interval_3 != interval_4


def test_combine():
    interval_1 = PreferenceInterval(interval={"A": 2, "B": 3, "C": 6, "D": 0})
    interval_2 = PreferenceInterval(interval={"A": 0.2, "B": 0.3, "C": 0.5})
    interval_3 = PreferenceInterval(interval={"E": 2, "F": 4})

    with pytest.raises(ValueError, match="Intervals must have disjoint candidate sets"):
        combine_preference_intervals([interval_1, interval_2], [1 / 2, 1 / 2])

    combined_pi = combine_preference_intervals([interval_1, interval_3], [1 / 3, 2 / 3])
    true_result = PreferenceInterval(
        interval={"A": 2 / 33, "B": 3 / 33, "C": 6 / 33, "D": 0, "E": 2 / 9, "F": 4 / 9}
    )

    assert combined_pi == true_result


def test_remove_zero():
    interval = PreferenceInterval(interval={"A": 2, "B": 3, "C": 6, "D": 0})

    # running a second time should do nothing
    interval._remove_zero_support_cands()

    assert interval.zero_cands == {"D"}
    assert interval.non_zero_cands == {"A", "B", "C"}
    assert interval.candidates == interval.non_zero_cands.union(interval.zero_cands)


def test_combine_triple():
    interval_1 = PreferenceInterval(interval={"A": 2, "B": 3, "C": 6, "D": 0})
    interval_2 = PreferenceInterval(interval={"A": 0.2, "B": 0.3, "C": 0.5})
    interval_3 = PreferenceInterval(interval={"E": 2, "F": 4})

    with pytest.raises(ValueError, match="Intervals must have disjoint candidate sets"):
        combine_preference_intervals(
            [interval_1, interval_2, interval_3], [1 / 3, 1 / 3, 1 / 3]
        )

    assert True


def test_combine_bad_proportions():
    interval_1 = PreferenceInterval(interval={"A": 2, "B": 3, "C": 6, "D": 0})
    interval_3 = PreferenceInterval(interval={"E": 2, "F": 4})

    # true_result = PreferenceInterval(interval={"A": 2/33, "B": 3/33, "C": 6/33,
    #                                                                "D":0, "E":2/9, "F":4/9})

    with pytest.raises(ValueError, match="Proportions must sum to 1."):
        combine_preference_intervals([interval_1, interval_3], [2 / 3, 4 / 3])
