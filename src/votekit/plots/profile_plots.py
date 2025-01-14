from matplotlib import pyplot as plt  # type: ignore
from votekit.pref_profile import PreferenceProfile
from votekit.utils import first_place_votes, mentions, COLOR_LIST, borda_scores
from matplotlib.axes import Axes
from typing import Optional, Union, Tuple, Literal, Callable
import matplotlib.patches as mpatches
from copy import deepcopy
import warnings
from matplotlib.legend import Legend


def add_null_keys(data: list[dict[str, float]]) -> list[dict[str, float]]:
    """
    Prepares list of dictionaries to be passed to ``bar_plot()``. If a key is missing from
    a dictionary, this function adds the key with value 0.

    Args:
        data (list[dict[str, float]]): Categorical data to be cleaned. The value of each dict
            should be the frequency of the key which is the category name.

    Returns:
        list[dict[str, float]]: Cleaned data.
    """
    data = deepcopy(data)

    x_labels = []
    for data_dict in data:
        for x_label in data_dict.keys():
            if x_label not in x_labels:
                x_labels.append(x_label)

    for x_label in x_labels:
        for data_dict in data:
            if x_label not in data_dict:
                data_dict[x_label] = 0

    return data


def _set_default_bar_plot_args(
    data: Union[list[dict[str, float]], dict[str, float]],
    data_set_labels: Optional[list[str]],
    data_set_to_color: Optional[dict],
    bar_width: Optional[float],
    x_label_ordering: Optional[list[str]],
    legend_font_size: Optional[float],
    ax: Optional[Axes],
) -> Tuple[
    list[dict[str, float]], list[str], dict, float, list[str], float, float, Axes
]:
    """
    Handles setting default arguments.

    Args:
        data (Union[list[dict[str, float]], dict[str, float]]): Categorical data to be plotted.
            If a list is provided, will plot all data sets on same plot. The value of the dict
            should be the frequency of the key which is the category name.
        data_set_labels (list[str], optional): List of labels for data sets.
        data_set_to_color (dict, optional): Dictionary mapping data set labels to colors. Defaults
            to subset of ``COLOR_LIST`` from ``utils`` module.
        bar_width (float, optional): Width of bars, defaults to None which computes the bar width
            as 0.7 divided by the number of data sets.
        x_label_ordering (list[str], optional): Ordering of x labels. Defaults to order retrieved
            from first data dictionary.
        ax (Axes, optional): A matplotlib axes object to plot the figure on. Defaults to None, in
            which case the function creates and returns a new axes.

    Returns:
        Tuple[list[dict[str, float]], list[str], dict, float, list[str], float, float, Axes]: data,
        data_set_labels, data_set_to_color, bar_width, x_label_ordering, FONT_SIZE,
        legend_font_size, ax
    """
    if not isinstance(data, list):
        data = [data]

    if not data_set_labels:
        data_set_labels = [f"Data Set {i+1}" for i in range(len(data))]

    if not data_set_to_color:
        data_set_to_color = {d: COLOR_LIST[i] for i, d in enumerate(data_set_labels)}

    if bar_width is None:
        bar_width = 0.7 / len(data)
    else:
        bar_width *= 1 / len(data)

    if not x_label_ordering:
        x_label_ordering = list(data[0].keys())

    FONT_SIZE = len(x_label_ordering) + 10

    if legend_font_size is None:
        legend_font_size = FONT_SIZE

    if ax is None:
        fig, ax = plt.subplots(figsize=(3 * len(x_label_ordering), 6))

    return (
        data,
        data_set_labels,
        data_set_to_color,
        bar_width,
        x_label_ordering,
        FONT_SIZE,
        legend_font_size,
        ax,
    )


def _validate_bar_plot_args(
    data: list[dict[str, float]],
    data_set_labels: list[str],
    x_label_ordering: list[str],
    bar_width: float,
):
    """
    Validates bar plot arguments. Raises ValueError.

    Args:
        data (list[dict[str, float]]): Categorical data to be plotted. The value of the dict
            should be the frequency of the key which is the category name.
        data_set_labels (list[str], optional): List of labels for data sets. Must be on per data
            set and must be unique.
        x_label_ordering (list[str]): Ordering of x labels. Must match data keys.
        bar_width (float): Width of bars. Must be between 0 and 1/len(data).

    """
    if any(set(data_dict.keys()) != set(data[0].keys()) for data_dict in data):
        raise ValueError("All data dictionaries must have the same keys.")

    if len(set(data_set_labels)) != len(data_set_labels):
        raise ValueError("Each data set must have a unique label.")

    if len(data_set_labels) != len(data):
        raise ValueError("There must be one data set label for each data set.")

    sym_dif = set(x_label_ordering) ^ set(
        k for data_dict in data for k in data_dict.keys()
    )
    if len(sym_dif) != 0:
        raise ValueError(
            (
                "x_label_ordering must match the keys of the data. Here is the symmetric "
                f"difference of the two sets: {sym_dif}"
            )
        )

    if bar_width <= 0:
        raise ValueError("Bar width must be positive.")

    elif bar_width > 1 / len(data):
        warnings.warn(
            "Bar width must be less than 1. Bar width has now been set to 1.",
            category=UserWarning,
        )
        bar_width = 1 / len(data)


def _normalize_data(data: dict[str, float]) -> dict[str, float]:
    """
    Normalizes data so number of total observations is 1. Raises a ValueError if the total
    mass is 0.

    Args:
        data (dict[str, float]]): Categorical data to be normalized. The value of the dict
            should be the frequency of the key which is the category name.ure on. Defaults to None,
            in which case the function creates and returns a new axes.

    Returns:
        dict[str, float]: Normalized data.
    """

    total_obs = sum(data.values())

    if total_obs == 0:
        raise ValueError("Total mass of observations must be non-zero.")

    return {k: v / total_obs for k, v in data.items()}


def _prepare_data_bar_plot(
    normalize: bool,
    data: list[dict[str, float]],
    x_label_ordering: list[str],
) -> list[list[float]]:
    """
    Formats data and normalizes if required.

    Args:
        normalize (bool): Whether or not to normalize data.
        data (list[dict[str, float]]): Categorical data to be plotted. The value of each dict
            should be the frequency of the key which is the category name.
        x_label_ordering (list[str]): Ordering of x labels.

    Returns:
        list[list[float]]: Height of bars, one list for each data set.

    """
    if normalize:
        data = [_normalize_data(data_dict) for data_dict in data]

    y_data = [
        [data_dict[x_label] for x_label in x_label_ordering] for data_dict in data
    ]

    return y_data


def _plot_datasets_on_bar_plot(
    ax: Axes,
    x_label_ordering: list[str],
    y_data: list[list[float]],
    data_set_labels: list[str],
    bar_width: float,
    data_set_to_color: dict,
    FONT_SIZE: float,
) -> Axes:
    """

    Args:
        x_label_ordering (list[str]): Ordering of x labels.
        y_data (list[list[floats]]): List of lists where each sublist is the bar heights for a
            data set.
        data_set_labels (list[str]): List of labels for data sets.
        bar_width (float): Width of bars.
        data_set_to_color (dict): Dictionary mapping data set labels to colors.
        FONT_SIZE (float): Font size for figure.

    Returns:
        Axes: Matplotlib axes containing bar plot.
    """

    for i, data_set in enumerate(y_data):
        bar_shift = bar_width * i - len(y_data) / 2 * bar_width
        bar_centers = [x + bar_shift for x in range(len(x_label_ordering))]

        align: Literal["center", "edge"] = "edge" if len(y_data) % 2 == 0 else "center"

        if i == len(y_data) // 2:
            ax.bar(
                bar_centers,
                data_set,
                tick_label=x_label_ordering,
                label=data_set_labels[i],
                align=align,
                width=bar_width,
                color=data_set_to_color[data_set_labels[i]],
            )
        else:
            ax.bar(
                bar_centers,
                data_set,
                label=data_set_labels[i],
                align=align,
                width=bar_width,
                color=data_set_to_color[data_set_labels[i]],
            )

    ax.tick_params(axis="x", labelsize=FONT_SIZE)
    return ax


def _label_bar_plot(
    ax: Axes,
    x_axis_name: Optional[str],
    y_axis_name: Optional[str],
    title: Optional[str],
    FONT_SIZE: float,
) -> Axes:
    """
    Add x,y axes labels and title to bar plot.

    Args:
        ax (Axes): Matplotlib axes containing barplot.
        x_axis_name (str, optional): Name of x-axis.
        y_axis_name (str, optional): Name of y-axis.
        title (str, optional): Title for the figure.
        FONT_SIZE (float): Font size for figure.

    Returns:
        Axes: Labeled bar plot.

    """
    if x_axis_name:
        ax.set_xlabel(x_axis_name, fontsize=FONT_SIZE)
    if y_axis_name:
        ax.set_ylabel(y_axis_name, fontsize=FONT_SIZE)
    if title:
        ax.set_title(title, fontsize=FONT_SIZE)

    return ax


def _add_data_sets_legend_bar_plot(
    ax: Axes,
    data_set_labels: list[str],
    data_set_to_color: dict,
    categories_legend: Optional[dict[str, str]],
    legend_font_size: float,
    legend_loc: str,
    legend_bbox_to_anchor: Tuple[float, float],
) -> Tuple[Axes, Legend]:
    """
    Add legend to bar plot for data sets.

    Args:
        ax (Axes): Matplotlib axes containing barplot.
        data_set_labels (list[str]): Labels for each data set.
        data_set_to_color (dict): Dictionary mapping data set labels to colors.
        categories_legend (dict[str, str], optional): Dictionary mapping x-axis
            categories to description in legend. Defaults to None, in which case legend is just
            x-axis labels.
        legend_font_size (float): The font size to use for the legend.
        legend_loc(str): The location parameter to pass to ``Axes.legend(loc=)``.
        legend_bbox_to_anchor (Tuple[float, float]): The bounding box to anchor the legend to.

    Returns:
        Tuple[Axes, Legend]: Matplotlib axes containing bar plot with legend for data sets.
    """
    proxy_artists = []

    for label in data_set_labels:
        patch = mpatches.Patch(color=data_set_to_color[label], label=label)
        proxy_artists.append(patch)

    if proxy_artists:
        leg = ax.legend(
            handles=proxy_artists,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            fontsize=legend_font_size,
            ncol=len(proxy_artists) // 15 + 1,
            frameon=True,
            fancybox=True,
        )

    if categories_legend:
        ax.add_artist(leg)

    return ax, leg


def _add_categories_legend_bar_plot(
    ax: Axes,
    categories_legend: dict[str, str],
    legend_font_size: float,
    legend_loc: str,
    legend_bbox_to_anchor: Tuple[float, float],
    data_set_legend: Optional[Legend],
) -> Axes:
    """
    Add legend to bar plot.

    Args:
        ax (Axes): Matplotlib axes containing barplot.
        categories_legend (dict[str, str]): Dictionary mapping x-axis
            categories to description in legend.
        legend_font_size (float): The font size to use for the legend.
        legend_loc(str): The location parameter to pass to ``Axes.legend(loc=)``.
        legend_bbox_to_anchor (Tuple[float, float]): The bounding box to anchor the legend to.
        data_set_legend (Legend, optional): The data set legend. Defaults to None.


    Returns:
        Axes: Matplotlib axes containing bar plot with legend.
    """
    proxy_artists = []

    for label, description in categories_legend.items():
        patch = mpatches.Patch(color="white", label=f"{label}: {description}")
        proxy_artists.append(patch)

    if proxy_artists:

        if data_set_legend:
            bbox = data_set_legend.get_window_extent()
            fig = plt.gcf()
            # fig.canvas.draw()

            width = bbox.transformed(fig.transFigure.inverted()).width
            legend_bbox_to_anchor = (
                legend_bbox_to_anchor[0] + 1.4 * width,
                legend_bbox_to_anchor[1],
            )

        ax.legend(
            handles=proxy_artists,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            fontsize=legend_font_size,
            ncol=len(proxy_artists) // 15 + 1,
            handlelength=0,
            handletextpad=0,
            frameon=True,
            fancybox=True,
        )

    return ax


def bar_plot(
    data: Union[list[dict[str, float]], dict[str, float]],
    *,
    data_set_labels: Optional[list[str]] = None,
    data_set_to_color: Optional[dict] = None,
    normalize: bool = False,
    bar_width: Optional[float] = None,
    x_label_ordering: Optional[list[str]] = None,
    x_axis_name: Optional[str] = None,
    y_axis_name: Optional[str] = None,
    title: Optional[str] = None,
    show_data_set_legend: bool = False,
    categories_legend: Optional[dict[str, str]] = None,
    legend_font_size: Optional[float] = None,
    legend_loc: str = "center left",
    legend_bbox_to_anchor: Tuple[float, float] = (1, 0.5),
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plots bar plot of categorical data.

    Args:
        data (Union[list[dict[str, float]], dict[str, float]]): Categorical data to be plotted.
            If a list of data sets is provided, will plot all data sets on same plot.
            The value of each dict should be the height of the bar corresponding to the key which
            is the category name.
        data_set_labels (list[str], optional): List of labels for data sets. Must be one per data
            set and must be unique. Defaults to "Data Set i" for each i.
        data_set_to_color (dict, optional): Dictionary mapping data set labels to colors. Defaults
            to subset of ``COLOR_LIST`` from ``utils`` module.
        normalize (bool, optional): Whether or not to normalize data. Defaults to False.
        bar_width (float, optional): Width of bars. Defaults to None which computes the bar width
            as 0.7 divided by the number of data sets. Must be in the interval :math:`(0,1]`.
        x_label_ordering (list[str], optional): Ordering of x-labels. Defaults to order retrieved
            from data dictionary.
        x_axis_name (str, optional): Name of x-axis. Defaults to None, which does not plot a name.
        y_axis_name (str, optional): Name of y-axis. Defaults to None, which does not plot a name.
        title (str, optional): Title for the figure. Defaults to None, which does not plot a title.
        show_data_set_legend (bool, optional): Whether or not to plot the data set legend.
            Defaults to False.
        categories_legend (dict[str, str], optional): Dictionary mapping data categories
            to description. Defaults to None. If provided, generates a second legend for data
            categories.
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
    (
        data,
        data_set_labels,
        data_set_to_color,
        bar_width,
        x_label_ordering,
        FONT_SIZE,
        legend_font_size,
        ax,
    ) = _set_default_bar_plot_args(
        data,
        data_set_labels,
        data_set_to_color,
        bar_width,
        x_label_ordering,
        legend_font_size,
        ax,
    )

    _validate_bar_plot_args(data, data_set_labels, x_label_ordering, bar_width)

    y_data = _prepare_data_bar_plot(normalize, deepcopy(data), x_label_ordering)

    ax = _plot_datasets_on_bar_plot(
        ax,
        x_label_ordering,
        y_data,
        data_set_labels,
        bar_width,
        data_set_to_color,
        FONT_SIZE,
    )

    ax = _label_bar_plot(ax, x_axis_name, y_axis_name, title, FONT_SIZE)

    data_set_legend = None
    if show_data_set_legend:
        ax, data_set_legend = _add_data_sets_legend_bar_plot(
            ax,
            data_set_labels,
            data_set_to_color,
            categories_legend,
            legend_font_size,
            legend_loc,
            legend_bbox_to_anchor,
        )

    if categories_legend:
        ax = _add_categories_legend_bar_plot(
            ax,
            categories_legend,
            legend_font_size,
            legend_loc,
            legend_bbox_to_anchor,
            data_set_legend,
        )

    return ax


def _set_default_args_plot_summary_stats(
    profile_list,
    stats_to_plot,
    stat_funcs_to_plot,
    profile_labels,
    threshold_value,
    threshold_label,
    candidate_ordering,
    use_integer_labels,
    candidate_legend,
):

    if not isinstance(profile_list, list):
        profile_list = [profile_list]

    if not isinstance(stats_to_plot, list):
        if stats_to_plot:
            stats_to_plot = [stats_to_plot]
        else:
            stats_to_plot = []

    for i, stat in enumerate(stats_to_plot):
        if stat == "fpv":
            stats_to_plot[i] = "first place votes"

    stats_to_func = {
        "first place votes": first_place_votes,
        "mentions": mentions,
        "borda": borda_scores,
    }

    if not stat_funcs_to_plot:
        stat_funcs_to_plot = {}

    stat_funcs_to_plot.update({stat: stats_to_func[stat] for stat in stats_to_plot})

    if not profile_labels:
        profile_labels = [f"Profile {i+1}" for i in range(len(profile_list))]

    if threshold_value and not threshold_label:
        threshold_label = f"Threshold: {threshold_value}"

    if not candidate_ordering:
        if len(stat_funcs_to_plot) == 1:
            sort_func = list(stat_funcs_to_plot.values())[0]
        else:
            sort_func = first_place_votes

        # default to reverse order
        sort_dict = sort_func(profile_list[0])
        candidate_ordering = sorted(
            sort_dict.keys(), key=lambda x: sort_dict[x], reverse=True
        )

    if not candidate_legend and use_integer_labels:
        candidate_legend = {i: c for i, c in enumerate(candidate_ordering)}
        candidate_ordering = list(candidate_legend.keys())

    return (
        profile_list,
        stat_funcs_to_plot,
        profile_labels,
        threshold_label,
        candidate_ordering,
        candidate_legend,
    )


def _validate_args_plot_summary_stats(profile_list, stat_funcs_to_plot):

    if any(set(p.candidates) != set(profile_list[0].candidates) for p in profile_list):
        raise ValueError("All PreferenceProfiles must have the same candidates.")

    if not stat_funcs_to_plot:
        raise ValueError(
            "At least one of stats_to_plot and stat_funcs_to_plot must be provided."
        )


def _prepare_data_plot_summary_stats(
    profile_list, stat_funcs_to_plot, profile_labels, candidate_legend
):
    # TODO this is where the order of bars within a candidate is decided
    data = [
        stat_func(profile)
        for profile in profile_list
        for stat_func in stat_funcs_to_plot.values()
    ]

    if candidate_legend:
        cand_to_label = {v: k for k, v in candidate_legend.items()}
        data = [
            {cand_to_label[c]: v for c, v in data_dict.items()} for data_dict in data
        ]

    else:
        data = [{c: v for c, v in data_dict.items()} for data_dict in data]

    data_set_labels = [
        f"{stat}, {label}" for stat in stat_funcs_to_plot for label in profile_labels
    ]

    return data, data_set_labels


def _add_threshold_line(
    threshold_value: float,
    threshold_color: str,
    threshold_line_style: str,
    threshold_line_width: float,
    threshold_label: str,
    ax: Axes,
):

    ax.axhline(
        threshold_value,
        linestyle=threshold_line_style,
        color=threshold_color,
        label=threshold_label,
        linewidth=threshold_line_width,
    )

    return ax


def plot_candidate_summary_stats(
    profile_list: Union[list[PreferenceProfile], PreferenceProfile],
    stats_to_plot: Optional[
        Union[
            list[Literal["fpv", "mentions", "borda", "first place votes"]],
            Literal["fpv", "mentions", "borda", "first place votes"],
        ]
    ] = None,
    stat_funcs_to_plot: Optional[
        dict[str, Callable]
    ] = None,  # if you want a custom function for score dict,
    profile_labels: Optional[list[str]] = None,
    normalize: bool = False,
    ax: Optional[Axes] = None,
    threshold_value: Optional[float] = None,
    threshold_color: str = "red",
    threshold_line_style: str = "--",
    threshold_line_width: float = 1,
    threshold_label: Optional[str] = None,
    candidate_ordering: Optional[list[str]] = None,
    use_integer_labels: bool = False,
    candidate_legend: Optional[dict[str, str]] = None,
    bar_width: Optional[float] = None,
    x_axis_name: Optional[str] = None,
    y_axis_name: Optional[str] = None,
    title: Optional[str] = None,
    show_stat_legend: bool = False,
    legend_font_size: Optional[float] = None,
    legend_loc: str = "center left",
    legend_bbox_to_anchor: Tuple[float, float] = (1, 0.5),
) -> Axes:
    """
    Plot a histogram of statistics about PreferenceProfiles. Has built in options for first-place
    votes, mentions, and Borda points. Allows users to choose a custom statistic.
    Allows users to plot multiple profiles and multiple stats on the same plot.

    Args:

    Returns:
        Axes: Matplotlib axes object with bar plot.
    """

    (
        profile_list,
        stat_funcs_to_plot,
        profile_labels,
        threshold_label,
        candidate_ordering,
        candidate_legend,
    ) = _set_default_args_plot_summary_stats(
        profile_list,
        stats_to_plot,
        stat_funcs_to_plot,
        profile_labels,
        threshold_value,
        threshold_label,
        candidate_ordering,
        use_integer_labels,
        candidate_legend,
    )

    _validate_args_plot_summary_stats(profile_list, stat_funcs_to_plot)

    data, data_set_labels = _prepare_data_plot_summary_stats(
        profile_list, stat_funcs_to_plot, profile_labels, candidate_legend
    )

    ax = bar_plot(
        data,
        data_set_labels=data_set_labels,
        normalize=normalize,
        bar_width=bar_width,
        x_label_ordering=candidate_ordering,
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name,
        title=title,
        show_data_set_legend=show_stat_legend,
        categories_legend=candidate_legend,
        legend_font_size=legend_font_size,
        legend_loc=legend_loc,
        legend_bbox_to_anchor=legend_bbox_to_anchor,
        ax=ax,
    )

    if threshold_value:
        # todo threshold and normalizing
        # threshold and profiles with different thresholds?
        ax = _add_threshold_line(
            threshold_value,
            threshold_color,
            threshold_line_style,
            threshold_line_width,
            threshold_label,
            ax,
        )

        # TODO add to legend
    return ax
