from matplotlib import pyplot as plt  # type: ignore
from votekit.pref_profile import PreferenceProfile
from votekit.utils import first_place_votes, mentions, COLOR_LIST, borda_scores
from matplotlib.axes import Axes
from typing import Optional, Union, Tuple, Literal, Callable
import matplotlib.patches as mpatches
from copy import deepcopy
import warnings
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

VALID_STATS = ["fpv", "mentions", "borda", "first place votes"]
Valid_stat = Literal["fpv", "mentions", "borda", "first place votes"]
DEFAULT_LINE_KWDS = {
    "ls": "-",
    "linewidth": 2,
}


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
    threshold_values: Optional[Union[list[float], float]],
    threshold_kwds: Optional[Union[list[dict], dict]],
    ax: Optional[Axes],
) -> Tuple[
    list[dict[str, float]],
    list[str],
    dict,
    float,
    list[str],
    float,
    float,
    Optional[list[float]],
    Optional[list[dict]],
    Axes,
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
        threshold_values (Union[list[float], float], optional): List of values to plot horizontal
            lines at. Can be provided as a list or a single float.
        threshold_kwds (Union[list[dict], dict], optional): List of plotting
            keywords for the horizontal lines. Can be a list or single dictionary.
        ax (Axes, optional): A matplotlib axes object to plot the figure on. Defaults to None, in
            which case the function creates and returns a new axes.

    Returns:
        Tuple[list[dict[str, float]], list[str], dict, float, list[str], float, float,
        Optional[list[float]], Optional[list[dict]], Axes]: data,
        data_set_labels, data_set_to_color, bar_width, x_label_ordering, FONT_SIZE,
        legend_font_size, threshold_values, threshold_kwds, ax
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

    FONT_SIZE = len(x_label_ordering) + 10.0

    if legend_font_size is None:
        legend_font_size = FONT_SIZE

    if threshold_values is not None and not isinstance(threshold_values, list):
        threshold_values = [threshold_values]

    if threshold_kwds is not None and not isinstance(threshold_kwds, list):
        threshold_kwds = [threshold_kwds]

    if isinstance(threshold_values, list):
        if not isinstance(threshold_kwds, list):
            threshold_kwds = [dict() for _ in range(len(threshold_values))]
        for i, kwd_dict in enumerate(threshold_kwds):
            if "color" not in kwd_dict:
                kwd_dict["color"] = COLOR_LIST[-(i + 1)]
            if "label" not in kwd_dict:
                kwd_dict["label"] = f"Line {i}"
            for k, v in DEFAULT_LINE_KWDS.items():
                if k not in kwd_dict:
                    kwd_dict[k] = v

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
        threshold_values,
        threshold_kwds,
        ax,
    )


def _validate_bar_plot_args(
    data: list[dict[str, float]],
    data_set_labels: list[str],
    x_label_ordering: list[str],
    bar_width: float,
    threshold_values: Optional[list[float]],
    threshold_kwds: Optional[list[dict]],
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
        threshold_values (list[float], optional): List of values to plot horizontal
            lines at.
        threshold_kwds (list[dict[str, str]], optional): List of plotting
            keywords for the horizontal lines.

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

    if threshold_values and threshold_kwds:
        if len(threshold_kwds) != len(threshold_values):
            raise ValueError(
                "threshold_values must have the same length as threshold_kwds."
            )


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


def _add_horizontal_lines_bar_plot(
    threshold_values: list[float],
    threshold_kwds: list[dict],
    ax: Axes,
):
    """
    Add horizontal lines to the bar plot.

    Args:
        threshold_values (list[float]): Display a horizontal line at given y-values.
        threshold_kwds (list[dict]): Matplotlib keywords for plotting lines.
        ax (Axes): Axes to plot on.

    Returns:
        Axes: Axes with horizontal lines added.

    """

    for i, y in enumerate(threshold_values):
        ax.axhline(y, **threshold_kwds[i])

    return ax


def _add_data_sets_legend_bar_plot(
    ax: Axes,
    data_set_labels: list[str],
    data_set_to_color: dict,
    categories_legend: Optional[dict[str, str]],
    threshold_kwds: Optional[list[dict]],
    legend_font_size: float,
    legend_loc: str,
    legend_bbox_to_anchor: Tuple[float, float],
) -> Tuple[Axes, Legend]:
    """
    Add legend to bar plot for data sets and any horizontal lines.

    Args:
        ax (Axes): Matplotlib axes containing barplot.
        data_set_labels (list[str]): Labels for each data set.
        data_set_to_color (dict): Dictionary mapping data set labels to colors.
        categories_legend (dict[str, str], optional): Dictionary mapping x-axis
            categories to description in legend. Defaults to None, in which case legend is just
            x-axis labels.
        threshold_kwds (list[dict], optional): List of plotting
            keywords for the horizontal lines.
        legend_font_size (float): The font size to use for the legend.
        legend_loc(str): The location parameter to pass to ``Axes.legend(loc=)``.
        legend_bbox_to_anchor (Tuple[float, float]): The bounding box to anchor the legend to.

    Returns:
        Tuple[Axes, Legend]: Matplotlib axes containing bar plot with legend for data sets.
    """
    proxy_artists: list[Union[mpatches.Patch, Line2D]] = []

    for label in data_set_labels:
        patch = mpatches.Patch(color=data_set_to_color[label], label=label)
        proxy_artists.append(patch)

    if threshold_kwds:
        for kwd_dict in threshold_kwds:
            line = Line2D([0], [0], **kwd_dict)
            proxy_artists.append(line)

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
    threshold_values: Optional[Union[list[float], float]] = None,
    threshold_kwds: Optional[Union[list[dict], dict]] = None,
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
        threshold_values (Union[list[float], float], optional): List of values to plot horizontal
            lines at. Can be provided as a list or a single float.
        threshold_kwds (Union[list[dict], dict], optional): List of plotting
            keywords for the horizontal lines. Can be a list or single dictionary.
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
        threshold_values,
        threshold_kwds,
        ax,
    ) = _set_default_bar_plot_args(
        data,
        data_set_labels,
        data_set_to_color,
        bar_width,
        x_label_ordering,
        legend_font_size,
        threshold_values,
        threshold_kwds,
        ax,
    )

    _validate_bar_plot_args(
        data,
        data_set_labels,
        x_label_ordering,
        bar_width,
        threshold_values,
        threshold_kwds,
    )

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

    if threshold_values and threshold_kwds:
        ax = _add_horizontal_lines_bar_plot(
            threshold_values,
            threshold_kwds,
            ax,
        )

    data_set_legend = None

    if show_data_set_legend:
        ax, data_set_legend = _add_data_sets_legend_bar_plot(
            ax,
            data_set_labels,
            data_set_to_color,
            categories_legend,
            threshold_kwds,
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
    profile_list: Union[list[PreferenceProfile], PreferenceProfile],
    stats_to_plot: Optional[Union[list[Valid_stat], Valid_stat]],
    stat_funcs_to_plot: Optional[dict[str, Callable]],
    profile_labels: Optional[list[str]],
    candidate_ordering: Optional[list[str]],
    use_integer_labels: bool,
    candidate_legend: Optional[dict[str, str]],
) -> Tuple[
    list[PreferenceProfile],
    dict[str, Callable],
    list[str],
    list[str],
    Optional[dict[str, str]],
    list[Valid_stat],
]:
    """
    Set the correct defaults for plotting.

    Args:
        profile_list (Union[list[PreferenceProfile], PreferenceProfile]): List of
            PreferenceProfiles to compute statistics for.
        stats_to_plot (Union[list[str], str], optional): Which statistic to plot. Must be one of
            ["fpv", "first place votes", "mentions", "borda"].
        stat_funcs_to_plot (dict[str, Callable], optional): Custom statistics to plot.
            Keys should be names of statistics, and values should be Callables. Callable should
            take a PreferenceProfile as input and return a dictionary whose keys are candidate names
            and values are the statistic.
        profile_labels (list[str], optional): List of labels for each profile.
        candidate_ordering (list[str], optional): Order in which to list candidates on the x-axis.
        use_integer_labels (bool): Use integer labels for candidates on x-axis.
        candidate_legend (dict[str, str, optional): Dictionary that maps x-axis labels to a
            description.

    Returns:
        Tuple[list[PreferenceProfile], dict[str, Callable], list[str], list[str],
        Optional[dict[str, str]], list[Valid_stat]]: profile_list, stat_funcs_to_plot,
        profile_labels, candidate_ordering, candidate_legend, stats_to_plot

    """

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

    try:
        stat_funcs_to_plot.update(
            {stat.capitalize(): stats_to_func[stat] for stat in stats_to_plot}
        )
    except KeyError as e:
        stat = e.args[0]
        raise ValueError(
            (
                f"{stat} is an invalid argument to stats_to_plot. "
                "Must be one of {VALID_STATS}."
            )
        )

    if not profile_labels:
        profile_labels = [f"Profile {i+1}" for i in range(len(profile_list))]

    if not candidate_ordering:
        sort_func = list(stat_funcs_to_plot.values())[0]

        # default to reverse order
        sort_dict = sort_func(profile_list[0])
        candidate_ordering = sorted(
            sort_dict.keys(), key=lambda x: sort_dict[x], reverse=True
        )

    if not candidate_legend and use_integer_labels:
        candidate_legend = {str(i): c for i, c in enumerate(candidate_ordering)}
        candidate_ordering = list(candidate_legend.keys())

    return (
        profile_list,
        stat_funcs_to_plot,
        profile_labels,
        candidate_ordering,
        candidate_legend,
        stats_to_plot,
    )


def _validate_args_plot_summary_stats(
    profile_list: list[PreferenceProfile],
    stat_funcs_to_plot: dict[str, Callable],
    candidate_ordering: list[str],
    candidate_legend: Optional[dict[str, str]],
    use_integer_labels: bool,
):
    """
    Validate and raise errors on arguments.

    Args:
        profile_list (list[PreferenceProfile]): List of
            PreferenceProfiles to compute statistics for.
        stat_funcs_to_plot (dict[str, Callable]): Custom statistics to plot.
            Keys should be names of statistics, and values should be Callables. Callable should
            take a PreferenceProfile as input and return a dictionary whose keys are candidate names
            and values are the statistic.
        candidate_ordering (list[str]): Order in which to list candidates on the x-axis.
        candidate_legend (dict[str, str, optional): Dictionary that maps x-axis labels to a
            description.
        use_integer_labels (bool): Use integer labels for candidates on x-axis.

    """

    if any(set(p.candidates) != set(profile_list[0].candidates) for p in profile_list):
        raise ValueError("All PreferenceProfiles must have the same candidates.")

    if not stat_funcs_to_plot:
        raise ValueError(
            "At least one of stats_to_plot and stat_funcs_to_plot must be provided."
        )

    if use_integer_labels and candidate_legend:
        sym_dif = set(candidate_legend.values()) ^ set(profile_list[0].candidates)

    else:
        sym_dif = set(candidate_ordering) ^ set(profile_list[0].candidates)

    if len(sym_dif) != 0:
        raise ValueError(
            (
                "Candidates listed in candidate_ordering must match the candidates in each profile."
                f" Here is the symmetric difference of the two sets: {sym_dif}"
            )
        )


def _prepare_data_plot_summary_stats(
    profile_list: list[PreferenceProfile],
    stat_funcs_to_plot: dict[str, Callable],
    profile_labels: list[str],
    candidate_legend: Optional[dict[str, str]],
) -> Tuple[list[dict[str, float]], list[str]]:
    """
    Convert PreferenceProfiles to correct dictionary data format.

    Args:
        profile_list (list[PreferenceProfile]): List of
            PreferenceProfiles to compute statistics for.
        stat_funcs_to_plot (dict[str, Callable]): Custom statistics to plot.
            Keys should be names of statistics, and values should be Callables. Callable should
            take a PreferenceProfile as input and return a dictionary whose keys are candidate names
            and values are the statistic.
        profile_labels (list[str]): List of labels for each profile.
        candidate_legend (dict[str, str], optional): Dictionary that maps x-axis labels to a
            description.

    Returns:
        Tuple[list[dict[str, float]], list[str]]: data, data_set_labels

    """
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


def plot_candidate_summary_stats(
    profile_list: Union[list[PreferenceProfile], PreferenceProfile],
    stats_to_plot: Optional[
        Union[
            list[Valid_stat],
            Valid_stat,
        ]
    ] = None,
    stat_funcs_to_plot: Optional[dict[str, Callable]] = None,
    profile_labels: Optional[list[str]] = None,
    show_stat_legend: bool = False,
    normalize: bool = False,
    threshold_values: Optional[Union[list[float], float]] = None,
    threshold_kwds: Optional[Union[list[dict], dict]] = None,
    candidate_ordering: Optional[list[str]] = None,
    use_integer_labels: bool = False,
    candidate_legend: Optional[dict[str, str]] = None,
    bar_width: Optional[float] = None,
    x_axis_name: Optional[str] = None,
    y_axis_name: Optional[str] = None,
    title: Optional[str] = None,
    legend_font_size: Optional[float] = None,
    legend_loc: str = "center left",
    legend_bbox_to_anchor: Tuple[float, float] = (1, 0.5),
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot a histogram of statistics about PreferenceProfiles. Has built in options for first-place
    votes, mentions, and Borda points. Allows users to choose a custom statistic.
    Allows users to plot multiple profiles and multiple stats on the same plot.

    Args:
        profile_list (Union[list[PreferenceProfile], PreferenceProfile]): List of
            PreferenceProfiles to compute statistics for. Can be provided as a single
            profile.
        stats_to_plot (Union[list[str],str], optional): Which statistic to plot. Can be either
            a list or a single stat. Must be one of ["fpv", "first place votes",
            "mentions", "borda"]. Defaults to None, in which case ``stat_funcs_to_plot`` is used.
        stat_funcs_to_plot (dict[str, Callable], optional): Custom statistics to plot.
            Keys should be names of statistics, and values should be Callables. Callable should
            take a PreferenceProfile as input and return a dictionary whose keys are candidate names
            and values are the statistic. Defaults to None, in which case ``stats_to_plot`` is used.
        profile_labels (list[str], optional): List of labels for each profile. Defaults to None,
            in which case labels default to "Profile i".
        show_stat_legend (bool, optional): Whether or not to show the legend containing profile
            and statistic labels. Defaults to False.
        normalize (bool, optional): Whether or not to normalize data. Defaults to False.
        threshold_values (Union[list[float], float], optional): List of values to plot horizontal
            lines at. Can be provided as a list or a single float.
        threshold_kwds (Union[list[dict], dict], optional): List of plotting
            keywords for the horizontal lines. Can be a list or single dictionary.
        candidate_ordering (list[str], optional): Order in which to list candidates on the x-axis.
            Defaults to None, in which case candidates are ordered in decreasing order of statistic.
            If multiple stats are listed, canonically chooses first statistic listed.
        use_integer_labels (bool, optional): Use integer labels for candidates on x-axis. Defaults
            to False. If True and ``candidate_legend`` is not provided, automatically draws a
            legend of the integer labels.
        candidate_legend (dict[str, str], optional): Dictionary that maps x-axis labels to a
            description. Defaults to None, in which case no candidate legend is displayed.
            If provided and ``use_integer_labels`` is True, this parameter overwrites the integer
            labeling legend.
        bar_width (float, optional): Width of bars. Defaults to None which computes the bar width
            as 0.7 divided by the number of data sets. Must be in the interval :math:`(0,1]`.
        x_axis_name (str, optional): Name of x-axis. Defaults to None, which does not plot a name.
        y_axis_name (str, optional): Name of y-axis. Defaults to None, which does not plot a name.
        title (str, optional): Title for the figure. Defaults to None, which does not plot a title.
        legend_font_size (float, optional): The font size to use for the legend. Defaults to 10.0
            + the number of categories.
        legend_loc (str, optional): The location parameter to pass to ``Axes.legend(loc=)``.
            Defaults to "center left".
        legend_bbox_to_anchor (Tuple[float, float], otptional): The bounding box to anchor
            the legend to. Defaults to (1, 0.5).
        ax (Axes, optional): A Matplotlib axes object to plot the figure on. Defaults to None, in
            which case the function creates and returns a new axes. The figure height is 6 inches
            and the figure width is 3 inches times the number of candidates.

    Returns:
        Axes: Matplotlib axes object with bar plot showing stats.
    """

    (
        profile_list,
        stat_funcs_to_plot,
        profile_labels,
        candidate_ordering,
        candidate_legend,
        stats_to_plot,
    ) = _set_default_args_plot_summary_stats(
        profile_list,
        stats_to_plot,
        stat_funcs_to_plot,
        profile_labels,
        candidate_ordering,
        use_integer_labels,
        candidate_legend,
    )

    _validate_args_plot_summary_stats(
        profile_list,
        stat_funcs_to_plot,
        candidate_ordering,
        candidate_legend,
        use_integer_labels,
    )

    data, data_set_labels = _prepare_data_plot_summary_stats(
        profile_list, stat_funcs_to_plot, profile_labels, candidate_legend
    )

    try:
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
            threshold_values=threshold_values,
            threshold_kwds=threshold_kwds,
            legend_font_size=legend_font_size,
            legend_loc=legend_loc,
            legend_bbox_to_anchor=legend_bbox_to_anchor,
            ax=ax,
        )

    except Exception as e:
        if e.args[0] == "There must be one data set label for each data set.":
            raise ValueError(
                "There must be exactly one profile label for each profile."
            )

    return ax
