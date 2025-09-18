from votekit.plots import (
    profile_borda_plot,
    profile_fpv_plot,
    profile_mentions_plot,
    profile_ballot_lengths_plot,
)
from votekit.pref_profile import PreferenceProfile
from votekit.ballot import Ballot
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

ballot_1 = Ballot(
    ranking=(frozenset({"Chris"}), frozenset({"Peter"}), frozenset({"Moon"})), weight=1
)
ballot_2 = Ballot(ranking=(frozenset({"Moon"}), frozenset({"Peter"})), weight=4)
ballot_3 = Ballot(ranking=(frozenset({"Chris"}),), weight=1)
ballot_4 = Ballot(ranking=(frozenset({"Peter"}),), weight=1)

profile = PreferenceProfile(ballots=(ballot_1, ballot_2, ballot_3, ballot_4))


def test_profile_borda_plot_with_defaults():
    ax = profile_borda_plot(profile)

    assert isinstance(ax, Axes)
    plt.close()


def test_profile_borda_plot_with_kwds():
    ax = profile_borda_plot(
        profile,
        profile_label="Profile Portland",
        normalize=True,
        profile_color="red",
        bar_width=1,
        candidate_ordering=["Chris", "Moon", "Peter"],
        x_axis_name="candidates",
        y_axis_name="fpv",
        title="plot",
        show_profile_legend=True,
        candidate_legend={"Chris": 5, "Peter": 6, "Moon": 7},
        relabel_candidates_with_int=True,
        threshold_values=1,
        threshold_kwds={"label": "threshold"},
        legend_font_size=4,
    )

    assert isinstance(ax, Axes)
    plt.close()


def test_profile_fpv_plot_with_defaults():
    ax = profile_fpv_plot(profile)

    assert isinstance(ax, Axes)
    plt.close()


def test_profile_fpv_plot_with_kwds():
    ax = profile_fpv_plot(
        profile,
        profile_label="Profile Portland",
        normalize=True,
        profile_color="red",
        bar_width=1,
        candidate_ordering=["Chris", "Moon", "Peter"],
        x_axis_name="candidates",
        y_axis_name="fpv",
        title="plot",
        show_profile_legend=True,
        candidate_legend={"Chris": 5, "Peter": 6, "Moon": 7},
        relabel_candidates_with_int=True,
        threshold_values=1,
        threshold_kwds={"label": "threshold"},
        legend_font_size=4,
    )

    assert isinstance(ax, Axes)
    plt.close()


def test_profile_mentions_plot_with_defaults():
    ax = profile_mentions_plot(profile)

    assert isinstance(ax, Axes)
    plt.close()


def test_profile_mentions_plot_with_kwds():
    ax = profile_mentions_plot(
        profile,
        profile_label="Profile Portland",
        normalize=True,
        profile_color="red",
        bar_width=1,
        candidate_ordering=["Chris", "Moon", "Peter"],
        x_axis_name="candidates",
        y_axis_name="fpv",
        title="plot",
        show_profile_legend=True,
        candidate_legend={"Chris": 5, "Peter": 6, "Moon": 7},
        relabel_candidates_with_int=True,
        threshold_values=1,
        threshold_kwds={"label": "threshold"},
        legend_font_size=4,
    )

    assert isinstance(ax, Axes)
    plt.close()


def test_profile_ballot_lengths_plot_with_defaults():
    ax = profile_ballot_lengths_plot(profile)

    assert isinstance(ax, Axes)
    plt.close()


def test_profile_ballot_lengths_plot_with_kwds():
    ax = profile_ballot_lengths_plot(
        profile,
        profile_label="Profile Portland",
        normalize=True,
        profile_color="red",
        bar_width=1,
        lengths_ordering=[3, 1, 2],
        x_axis_name="lengths",
        y_axis_name="fpv",
        title="plot",
        show_profile_legend=True,
        lengths_legend={1: 5, 2: 6, 3: 7},
        threshold_values=1,
        threshold_kwds={"label": "threshold"},
        legend_font_size=4,
    )

    assert isinstance(ax, Axes)
    plt.close()
