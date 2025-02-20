from typing import Callable, Optional, Union, Any
from ...pref_profile import PreferenceProfile
from matplotlib.axes import Axes
from ...utils import (
    first_place_votes,
    borda_scores,
    mentions,
    ballot_lengths,
    COLOR_LIST,
)
from functools import partial
from ..bar_plot import add_null_keys, multi_bar_plot

plot_functions = {
    "first place votes": first_place_votes,
    "mentions": mentions,
    "borda": borda_scores,
    "ballot length": ballot_lengths,
}


def _partial_stat_function(
    stat_function: Union[str, Callable[[PreferenceProfile], dict[str, float]]],
    stat_function_kwds: Optional[dict],
):
    """
    Takes the stat function string and returns the corresponding callable with the correct
    keywords.

    Args:
        stat_function (Union[str, Callable[[PreferenceProfile], dict[str, float]]]): Which stat
            to use for the bar plot. Can be one of "first place votes", "mentions", "borda", or
            "ballot_length", or it can be a custom callable that takes a profile and returns
            a dict with str keys and float values.
        stat_function_kwds (dict[str, Any], optional): Keyword arguments to pass to
            ``stat_function``.
    """
    if isinstance(stat_function, str):
        stat_function = plot_functions[stat_function]  # type: ignore[assignment]

        if stat_function_kwds and callable(stat_function):
            stat_function = partial(stat_function, **stat_function_kwds)

    return stat_function


def _validate_inputs(
    stat_function: Union[str, Callable[[PreferenceProfile], dict[str, float]]]
):
    """
    Validate inputs to bar plot.

    Args:
        stat_function (Union[str, Callable[[PreferenceProfile], dict[str, float]]]): Which stat
            to use for the bar plot. Can be one of "first place votes", "mentions", "borda", or
            "ballot_length", or it can be a custom callable that takes a profile and returns
            a dict with str keys and float values.

    Raises:
        ValueError: if stat_function str is not one of the defined stats.
    """

    if isinstance(stat_function, str):
        if stat_function not in plot_functions:
            raise ValueError(
                (
                    f"stat_function string {stat_function} not an available statistic."
                    f" Available stats include: {list(plot_functions.keys())}"
                )
            )


def profile_multi_bar_plot(
    profile_dict: dict[str, PreferenceProfile],
    stat_function: Union[str, Callable[[PreferenceProfile], dict[str, float]]],
    stat_function_kwds: Optional[dict[str, Any]] = None,
    normalize: bool = False,
    profile_colors: Optional[dict[str, str]] = None,
    bar_width: Optional[float] = None,
    category_ordering: Optional[list[str]] = None,
    x_axis_name: Optional[str] = None,
    y_axis_name: Optional[str] = None,
    title: Optional[str] = None,
    show_profile_legend: bool = False,
    categories_legend: Optional[dict[str, str]] = None,
    threshold_values: Optional[Union[list[float], float]] = None,
    threshold_kwds: Optional[Union[list[dict], dict]] = None,
    legend_font_size: Optional[float] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot a multi bar plot over a set of preference profiles and some statistic about the profile,
    like ballot length, first place votes for candidate, etc.

    Args:
        profile_dict (dict[str, PreferenceProfile]): Keys are profile labels and values are
            profiles to plot statistics for.
        stat_function (Union[str, Callable[[PreferenceProfile], dict[str, float]]]): Which stat
            to use for the bar plot. Can be one of "first place votes", "mentions", "borda", or
            "ballot_length", or it can be a custom callable that takes a profile and returns
            a dict with str keys and float values.
        stat_function_kwds (dict[str, Any], optional): Keyword arguments to pass to
            ``stat_function``. Defaults to None, in which case default values for ``stat_function``
            are used.
        normalize (bool, optional): Whether or not to normalize data. Defaults to False.
        profile_colors (dict[str, str], optional): Dictionary mapping profile labels
            to colors. Defaults to None, in which case we use a subset of ``COLOR_LIST``
            from ``utils`` module. Dictionary keys can be a subset of the profiles.
        bar_width (float, optional): Width of bars. Defaults to None which computes the bar width
            as 0.7 divided by the number of data sets. Must be in the interval :math:`(0,1]`.
        category_ordering (list[str], optional): Ordering of x-labels. Defaults to order retrieved
            from data dictionary.
        x_axis_name (str, optional): Name of x-axis. Defaults to None, which does not plot a name.
        y_axis_name (str, optional): Name of y-axis. Defaults to None, which does not plot a name.
        title (str, optional): Title for the figure. Defaults to None, which does not plot a title.
        show_profile_legend (bool, optional): Whether or not to plot the profile legend.
            Defaults to False. Is automatically shown if any threshold lines have the keyword
            "label" passed through ``threshold_kwds``.
        categories_legend (dict[str, str], optional): Dictionary mapping data categories
            to description. Defaults to None. If provided, generates a second legend for data
            categories.
        threshold_values (Union[list[float], float], optional): List of values to plot horizontal
            lines at. Can be provided as a list or a single float.
        threshold_kwds (Union[list[dict], dict], optional): List of plotting
            keywords for the horizontal lines. Can be a list or single dictionary. These will be
            passed to plt.axhline(). Common keywords include "linestyle", "linewidth", and "label".
            If "label" is passed, automatically plots the data set legend with the labels.
        legend_font_size (float, optional): The font size to use for the legend. Defaults to 10.0
            + the number of categories.
        legend_loc (str, optional): The location parameter to pass to ``Axes.legend(loc=)``.
            Defaults to "center left".
        legend_bbox_to_anchor (Tuple[float, float], otptional): The bounding box to anchor
            the legend to. Defaults to (1, 0.5).
        ax (Axes, optional): A matplotlib axes object to plot the figure on. Defaults to None, in
            which case the function creates and returns a new axes. The figure height is 6 inches
            and the figure width is 3 inches times the number of categories.

    Returns:
        Axes: A ``matplotlib`` axes with a bar plot of the given data.
    """

    _validate_inputs(stat_function=stat_function)

    stat_function = _partial_stat_function(
        stat_function=stat_function, stat_function_kwds=stat_function_kwds
    )

    if callable(stat_function):
        data_dict = {
            label: stat_function(profile) for label, profile in profile_dict.items()
        }
        data_dict = add_null_keys(data_dict)

    try:
        ax = multi_bar_plot(
            data_dict,
            normalize=normalize,
            data_set_colors=profile_colors,
            bar_width=bar_width,
            category_ordering=category_ordering,
            x_axis_name=x_axis_name,
            y_axis_name=y_axis_name,
            title=title,
            show_data_set_legend=show_profile_legend,
            categories_legend=categories_legend,
            threshold_values=threshold_values,
            threshold_kwds=threshold_kwds,
            legend_font_size=legend_font_size,
            ax=ax,
        )

        return ax
    except Exception as e:
        if "Cannot plot more than" in str(e):
            raise ValueError(f"Cannot plot more than {len(COLOR_LIST)} profiles.")

        elif str(e) == "category_ordering must be the same length as sub-dictionaries.":
            raise ValueError(
                (
                    "category_ordering must be the same length as the dictionary "
                    f"produced by stat_function: {len(next(iter(data_dict.values())))}."
                )
            )

        else:
            raise e


# TODO
def profile_bar_plot():
    pass
