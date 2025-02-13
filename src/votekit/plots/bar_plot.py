from matplotlib import pyplot as plt  # type: ignore
from votekit.utils import COLOR_LIST
from matplotlib.axes import Axes
from typing import Optional, Union, Tuple, Literal, Any
import matplotlib.patches as mpatches
import warnings
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

VALID_STATS = ["fpv", "mentions", "borda", "first place votes"]
Valid_stat = Literal["fpv", "mentions", "borda", "first place votes"]
DEFAULT_LINE_KWDS = {
    "linestyle": "-",
    "linewidth": 2,
}


def add_null_keys(data: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """
    Prepares dictionary of dictionaries to be passed to ``bar_plot()``. If a key is missing from
    a dictionary, this function adds the key with value 0.

    Args:
        data (dict[str, dict[str, float]]): Categorical data to be cleaned. The value of each dict
            should be the frequency of the key which is the category name.

    Returns:
        dict[str, dict[str, float]]: Cleaned data.
    """

    x_labels = []
    for data_dict in data.values():
        for x_label in data_dict.keys():
            if x_label not in x_labels:
                x_labels.append(x_label)

    for x_label in x_labels:
        for data_dict in data.values():
            if x_label not in data_dict:
                data_dict[x_label] = 0

    return data


def _set_default_bar_plot_args(
    *,
    data: dict[str, dict[str, float]],
    data_set_colors: Optional[dict],
    bar_width: Optional[float],
    category_ordering: Optional[list[str]],
    legend_font_size: Optional[float],
    threshold_values: Optional[Union[list[float], float]],
    threshold_kwds: Optional[Union[list[dict], dict]],
    ax: Optional[Axes],
) -> dict[str, Any]:
    """
    Handles setting default arguments.

    Args:
        data (dict[str, dict[str, float]]): Categorical data to be plotted. Top level keys are
            data set labels. Inner keys are categories, and inner values are the height of the bars.
        data_set_colors (dict, optional): Dictionary mapping data set labels or data set indices to
            colors. Defaults to None, in which case we use a subset of ``COLOR_LIST`` from ``utils``
            module. Dictionary keys can be a subset of the data sets.
        bar_width (float, optional): Width of bars, defaults to None which computes the bar width
            as 0.7 divided by the number of data sets.
        category_ordering (list[str], optional): Ordering of x labels. Defaults to order retrieved
            from first data dictionary.
        threshold_values (Union[list[float], float], optional): List of values to plot horizontal
            lines at. Can be provided as a list of values or a single float.
        threshold_kwds (Union[list[dict], dict], optional): List of plotting
            keywords for the horizontal lines. Can be a list of dictionaries or single dictionary.
        ax (Axes, optional): A matplotlib axes object to plot the figure on. Defaults to None, in
            which case the function creates and returns a new axes.

    Returns:
        dict[str, Any]: data_set_to_color, bar_width, category_ordering, font_size,
        legend_font_size, threshold_values, threshold_kwds, ax
    """

    data_set_to_color = {d: COLOR_LIST[i] for i, d in enumerate(data.keys())}

    if data_set_colors:
        for data_set, color in data_set_colors.items():
            data_set_to_color[data_set] = color

    if bar_width is None:
        bar_width = 0.7 / len(data)
    else:
        bar_width *= 1 / len(data)

    if not category_ordering:
        category_ordering = list(next(iter(data.values())).keys())

    FONT_SIZE = len(category_ordering) + 10.0

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
                kwd_dict["label"] = f"Line {i+1}"
            for k, v in DEFAULT_LINE_KWDS.items():
                if k not in kwd_dict:
                    kwd_dict[k] = v

    if ax is None:
        fig, ax = plt.subplots(figsize=(3 * len(category_ordering), 6))

    return {
        "data_set_to_color": data_set_to_color,
        "bar_width": bar_width,
        "category_ordering": category_ordering,
        "font_size": FONT_SIZE,
        "legend_font_size": legend_font_size,
        "threshold_values": threshold_values,
        "threshold_kwds": threshold_kwds,
        "ax": ax,
    }


def _validate_bar_plot_args(
    *,
    data: dict[str, dict[str, float]],
    category_ordering: list[str],
    bar_width: float,
    threshold_values: Optional[list[float]],
    threshold_kwds: Optional[list[dict[str, str]]],
) -> float:
    """
    Validates bar plot arguments.

    Args:
        data (dict[str, dict[str, float]]): Categorical data to be plotted. Top level keys are
            data set labels. Inner keys are categories, and inner values are the height of the bars.
        category_ordering (list[str]): Ordering of x labels. Must match data keys.
        bar_width (float): Width of bars. Must be between 0 and 1/len(data).
        threshold_values (list[float], optional): List of values to plot horizontal
            lines at.
        threshold_kwds (list[dict[str, str]], optional): List of plotting
            keywords for the horizontal lines.

    Raises:
        ValueError: All bar heights must be positive.
        ValueError: All sub dictionaries in data must have the same keys.
        ValueError: category_ordering must match the keys of data (unordered).
        ValueError: Bar width must be positive.
        UserWarning: Bar width must be less than 1.
        ValueError: threshold_values and kwds must have the same length.
        ValueError: If lw or ls is passed as a threshold keyword.

    Returns:
        float: Bar width.
    """
    if any(v < 0 for data_dict in data.values() for v in data_dict.values()):
        raise ValueError("All bar heights must be non-negative.")

    if any(
        set(next(iter(data.values())).keys()) != set(data_dict.keys())
        for data_dict in data.values()
    ):
        raise ValueError("All data dictionaries must have the same categorical keys.")

    sym_dif = set(category_ordering) ^ set(next(iter(data.values())).keys())
    if len(sym_dif) != 0:
        raise ValueError(
            (
                "category_ordering must match the keys of the data sub-dictionaries. "
                "Here is the symmetric "
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
        if any("ls" in d.keys() for d in threshold_kwds):
            raise ValueError("Must use linestyle, not ls.")
        if any("lw" in d.keys() for d in threshold_kwds):
            raise ValueError("Must use linewidth, not lw.")

    return bar_width


def _normalize_data_dict(data_dict: dict[str, float]) -> dict[str, float]:
    """
    Normalizes data so number of total observations is 1. Raises a ValueError if the total
    mass is 0.

    Args:
        data (dict[str, float]): Single data dictionary whose keys are categories and values
            are bar heights.

    Returns:
        dict[str, float]: Normalized data.

    Raise:
        ValueError: If the sum of the data is 0.
    """

    total_obs = sum(data_dict.values())

    if total_obs == 0:
        raise ValueError("Total mass of observations must be non-zero.")

    return {k: v / total_obs for k, v in data_dict.items()}


def _prepare_data_bar_plot(
    *,
    normalize: bool,
    data: dict[str, dict[str, float]],
    category_ordering: list[str],
) -> list[list[float]]:
    """
    Formats data and normalizes if required.

    Args:
        normalize (bool): Whether or not to normalize data.
        data (dict[str, dict[str, float]]): Categorical data to be plotted. Top level keys are
            data set labels. Inner keys are categories, and inner values are the height of the bars.
        category_ordering (list[str]): Ordering of x labels.

    Returns:
        list[list[float]]: Height of bars, one list for each data set.

    """
    if normalize:
        data = {
            label: _normalize_data_dict(data_dict) for label, data_dict in data.items()
        }

    y_data = [
        [data_dict[x_label] for x_label in category_ordering]
        for data_dict in data.values()
    ]

    return y_data


def _plot_datasets_on_bar_plot(
    *,
    ax: Axes,
    category_ordering: list[str],
    y_data: list[list[float]],
    data_set_labels: list[str],
    bar_width: float,
    data_set_to_color: dict,
    font_size: float,
) -> Axes:
    """

    Args:
        category_ordering (list[str]): Ordering of x labels.
        y_data (list[list[floats]]): List of lists where each sublist is the bar heights for a
            data set.
        data_set_labels (list[str]): List of labels for data sets.
        bar_width (float): Width of bars.
        data_set_to_color (dict): Dictionary mapping data set labels to colors.
        font_size (float): Font size for figure.

    Returns:
        Axes: Matplotlib axes containing bar plot.
    """

    for i, data_set in enumerate(y_data):
        bar_shift = bar_width * i - len(y_data) / 2 * bar_width
        bar_centers = [x + bar_shift for x in range(len(category_ordering))]

        align: Literal["center", "edge"] = "edge" if len(y_data) % 2 == 0 else "center"

        if i == len(y_data) // 2:
            ax.bar(
                bar_centers,
                data_set,
                tick_label=category_ordering,
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

    ax.tick_params(axis="x", labelsize=font_size)
    return ax


def _label_bar_plot(
    *,
    ax: Axes,
    x_axis_name: Optional[str],
    y_axis_name: Optional[str],
    title: Optional[str],
    font_size: float,
) -> Axes:
    """
    Add x,y axes labels and title to bar plot.

    Args:
        ax (Axes): Matplotlib axes containing barplot.
        x_axis_name (str, optional): Name of x-axis.
        y_axis_name (str, optional): Name of y-axis.
        title (str, optional): Title for the figure.
        font_size (float): Font size for figure.

    Returns:
        Axes: Labeled bar plot.

    """
    if x_axis_name:
        ax.set_xlabel(x_axis_name, fontsize=font_size)
    if y_axis_name:
        ax.set_ylabel(y_axis_name, fontsize=font_size)
    if title:
        ax.set_title(title, fontsize=font_size)

    return ax


def _add_horizontal_lines_bar_plot(
    *,
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
    *,
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
    *,
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
    data: dict[str, dict[str, float]],
    *,
    normalize: bool = False,
    data_set_colors: Optional[dict[str, str]] = None,
    bar_width: Optional[float] = None,
    category_ordering: Optional[list[str]] = None,
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
        data (dict[str, dict[str, float]]): Categorical data to be plotted. Top level keys are
            data set labels. Inner keys are categories, and inner values are the height of the bars.
        normalize (bool, optional): Whether or not to normalize data. Defaults to False.
        data_set_colors (dict[str, str], optional): Dictionary mapping data set labels
            to colors. Defaults to None, in which case we use a subset of ``COLOR_LIST``
            from ``utils`` module. Dictionary keys can be a subset of the data sets.
        bar_width (float, optional): Width of bars. Defaults to None which computes the bar width
            as 0.7 divided by the number of data sets. Must be in the interval :math:`(0,1]`.
        category_ordering (list[str], optional): Ordering of x-labels. Defaults to order retrieved
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

    default_values_dict = _set_default_bar_plot_args(
        data=data,
        data_set_colors=data_set_colors,
        bar_width=bar_width,
        category_ordering=category_ordering,
        legend_font_size=legend_font_size,
        threshold_values=threshold_values,
        threshold_kwds=threshold_kwds,
        ax=ax,
    )

    bar_width = _validate_bar_plot_args(
        data=data,
        category_ordering=default_values_dict["category_ordering"],
        bar_width=default_values_dict["bar_width"],
        threshold_values=default_values_dict["threshold_values"],
        threshold_kwds=default_values_dict["threshold_kwds"],
    )

    y_data = _prepare_data_bar_plot(
        normalize=normalize,
        data=data,
        category_ordering=default_values_dict["category_ordering"],
    )

    ax = _plot_datasets_on_bar_plot(
        ax=default_values_dict["ax"],
        category_ordering=default_values_dict["category_ordering"],
        y_data=y_data,
        data_set_labels=list(data.keys()),
        bar_width=default_values_dict["bar_width"],
        data_set_to_color=default_values_dict["data_set_to_color"],
        font_size=default_values_dict["font_size"],
    )

    ax = _label_bar_plot(
        ax=ax,
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name,
        title=title,
        font_size=default_values_dict["font_size"],
    )

    if (
        default_values_dict["threshold_values"]
        and default_values_dict["threshold_kwds"]
    ):
        ax = _add_horizontal_lines_bar_plot(
            threshold_values=default_values_dict["threshold_values"],
            threshold_kwds=default_values_dict["threshold_kwds"],
            ax=ax,
        )

    data_set_legend = None

    if show_data_set_legend:
        ax, data_set_legend = _add_data_sets_legend_bar_plot(
            ax=ax,  # type: ignore[arg-type]
            data_set_labels=list(data.keys()),
            data_set_to_color=default_values_dict["data_set_to_color"],
            categories_legend=categories_legend,
            threshold_kwds=default_values_dict["threshold_kwds"],
            legend_font_size=default_values_dict["legend_font_size"],
            legend_loc=legend_loc,
            legend_bbox_to_anchor=legend_bbox_to_anchor,
        )

    if categories_legend:
        ax = _add_categories_legend_bar_plot(
            ax=ax,  # type: ignore[arg-type]
            categories_legend=categories_legend,
            legend_font_size=default_values_dict["legend_font_size"],
            legend_loc=legend_loc,
            legend_bbox_to_anchor=legend_bbox_to_anchor,
            data_set_legend=data_set_legend,
        )

    return ax  # type: ignore[return-value]
