from votekit.plots.profile_plots import (
    _set_default_args_plot_summary_stats,
    _validate_args_plot_summary_stats,
    _prepare_data_plot_summary_stats,
    plot_candidate_summary_stats,
)
from votekit import PreferenceProfile, Ballot
from votekit.utils import first_place_votes, borda_scores
import pytest
from matplotlib.axes import Axes


ballot_1 = Ballot(
    ranking=(frozenset({"Chris"}), frozenset({"Peter"}), frozenset({"Moon"})), weight=1
)
ballot_2 = Ballot(ranking=(frozenset({"Moon"}), frozenset({"Peter"})), weight=4)
ballot_3 = Ballot(ranking=(frozenset({"Chris"}),), weight=1)
ballot_4 = Ballot(ranking=(frozenset({"Peter"}),), weight=1)

profile_1 = PreferenceProfile(ballots=(ballot_1, ballot_2, ballot_3))
profile_2 = PreferenceProfile(ballots=(ballot_1, ballot_2, ballot_4))
profile_3 = PreferenceProfile(ballots=(ballot_4,))


def test_set_default_args_none():

    (
        profile_list,
        stat_funcs_to_plot,
        profile_labels,
        candidate_ordering,
        candidate_legend,
        stats_to_plot,
    ) = _set_default_args_plot_summary_stats(
        profile_1, None, None, None, None, False, None
    )

    assert isinstance(profile_list, list)
    assert profile_list[0] == profile_1
    assert stat_funcs_to_plot == {"first place votes": first_place_votes}
    assert profile_labels == ["Profile 1"]
    assert candidate_ordering == ["Moon", "Chris", "Peter"]
    assert candidate_legend is None
    assert isinstance(stats_to_plot, list)
    assert stats_to_plot == []


def test_set_default_args_provided():

    (
        profile_list,
        stat_funcs_to_plot,
        profile_labels,
        candidate_ordering,
        candidate_legend,
        stats_to_plot,
    ) = _set_default_args_plot_summary_stats(
        [profile_1, profile_2],
        "borda",
        {"test_func": first_place_votes},
        ["p1", "p2"],
        ["Chris", "Peter", "Moon"],
        True,
        {"0": "hi", "1": "yo", "2": "yep"},
    )

    assert isinstance(profile_list, list)
    assert profile_list == [profile_1, profile_2]
    assert stat_funcs_to_plot == {"test_func": first_place_votes, "borda": borda_scores}
    assert profile_labels == ["p1", "p2"]
    assert candidate_ordering == ["Chris", "Peter", "Moon"]
    assert candidate_legend == {"0": "hi", "1": "yo", "2": "yep"}
    assert stats_to_plot == ["borda"]


def test_validate_args_errors():
    with pytest.raises(
        ValueError, match="All PreferenceProfiles must have the same candidates."
    ):
        _validate_args_plot_summary_stats(
            [profile_1, profile_3], ["Chris", "Peter"], None, False
        )

    with pytest.raises(
        ValueError,
        match="Candidates listed in candidate_ordering must match the candidates in each profile.",
    ):

        _validate_args_plot_summary_stats(
            [profile_1], ["Chris", "Paul", "Moon"], None, False
        )


def test_prepare_data():
    data, data_set_labels = _prepare_data_plot_summary_stats(
        [profile_1],
        {"first place votes": first_place_votes},
        ["p1"],
        None,
    )

    assert data[0] == {"Chris": 2, "Peter": 0, "Moon": 4}

    assert data_set_labels == ["First place votes, p1"]


def test_simple_plot():
    ax = plot_candidate_summary_stats(profile_1, "mentions")

    assert isinstance(ax, Axes)


def test_multi_plot():
    ax = plot_candidate_summary_stats(
        [profile_1, profile_2],
        "mentions",
        {"fpv": first_place_votes},
        ["p1", "p2"],
        True,
        True,
        [5],
        None,
        None,
        True,
        None,
    )

    assert isinstance(ax, Axes)
