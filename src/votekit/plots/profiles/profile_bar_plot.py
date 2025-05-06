from typing import Callable, Optional, Union, Any
from ...pref_profile import PreferenceProfile
from .multi_profile_bar_plot import multi_profile_bar_plot
from ..bar_plot import bar_plot
from matplotlib.axes import Axes
from ...utils import (
    first_place_votes,
    borda_scores,
    mentions,
    ballot_lengths,
    COLOR_LIST,
)


def profile_bar_plot(
    profile: PreferenceProfile,
    stat_function: Callable[[PreferenceProfile], dict[str, float]],
    *,
    profile_label: str = "Profile",
    normalize: bool = False,
    profile_color: str = COLOR_LIST[0],
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
    Plots bar plot of single profile. Wrapper for ``multi_profile_bar_plot``.

    Args:
        profile (PreferenceProfile): Profile to plot statistics for.
        stat_function (Callable[[PreferenceProfile], dict[str, float]]): Which stat
            to use for the bar plot. Must be a callable that takes a profile and returns
            a dict with str keys and float values.
        profile_label (str, optional): Label for profile. Defaults to "Profile".
        normalize (bool, optional): Whether or not to normalize data. Defaults to False.
        profile_color (str, optional): Color to plot. Defaults to the first color from
            ``COLOR_LIST`` from ``utils`` module.
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

    return multi_profile_bar_plot(
        {profile_label: profile},
        stat_function=stat_function,
        normalize=normalize,
        profile_colors={profile_label: profile_color},
        bar_width=bar_width,
        category_ordering=category_ordering,
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name,
        title=title,
        show_profile_legend=show_profile_legend,
        categories_legend=categories_legend,
        threshold_values=threshold_values,
        threshold_kwds=threshold_kwds,
        legend_font_size=legend_font_size,
        ax=ax,
    )


def profile_borda_plot(
    profile: PreferenceProfile,
    *,
    profile_label: str = "Profile",
    borda_kwds: Optional[dict[str, Any]] = None,
    normalize: bool = False,
    profile_color: str = COLOR_LIST[0],
    bar_width: Optional[float] = None,
    candidate_ordering: Optional[list[str]] = None,
    x_axis_name: Optional[str] = None,
    y_axis_name: Optional[str] = None,
    title: Optional[str] = None,
    show_profile_legend: bool = False,
    candidate_legend: Optional[dict[str, str]] = None,
    relabel_candidates_with_int: bool = False,
    threshold_values: Optional[Union[list[float], float]] = None,
    threshold_kwds: Optional[Union[list[dict], dict]] = None,
    legend_font_size: Optional[float] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plots borda points of candidates in profile. Wrapper for ``profile_bar_plot``.

    Args:
        profile (PreferenceProfile): Profile to plot statistics for.
        profile_label (str, optional): Label for profile. Defaults to "Profile".
        borda_kwds (dict[str, Any], optional): Keyword arguments to pass to
            ``borda_scores``. Defaults to None, in which case default values for ``borda_scores``
            are used.
        normalize (bool, optional): Whether or not to normalize data. Defaults to False.
        profile_color (str, optional): Color to plot. Defaults to the first color from
            ``COLOR_LIST`` from ``utils`` module.
        bar_width (float, optional): Width of bars. Defaults to None which computes the bar width
            as 0.7 divided by the number of data sets. Must be in the interval :math:`(0,1]`.
        candidate_ordering (list[str], optional): Ordering of x-labels. Defaults to decreasing
            order of Borda scores.
        x_axis_name (str, optional): Name of x-axis. Defaults to None, which does not plot a name.
        y_axis_name (str, optional): Name of y-axis. Defaults to None, which does not plot a name.
        title (str, optional): Title for the figure. Defaults to None, which does not plot a title.
        show_profile_legend (bool, optional): Whether or not to plot the profile legend.
            Defaults to False. Is automatically shown if any threshold lines have the keyword
            "label" passed through ``threshold_kwds``.
        candidate_legend (dict[str, str], optional): Dictionary mapping candidates
            to alternate label. Defaults to None. If provided, generates a second legend.
        relabel_candidates_with_int (bool, optional): Relabel the candidates with integer labels.
            Defaults to False. If ``candidate_legend`` is passed, those labels supercede.
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
    score_dict = (
        borda_scores(profile, **borda_kwds) if borda_kwds else borda_scores(profile)
    )

    if candidate_ordering is None:
        candidate_ordering = sorted(
            score_dict.keys(), reverse=True, key=lambda x: score_dict[x]
        )

    if relabel_candidates_with_int and not candidate_legend:
        candidate_legend = {c: str(i + 1) for i, c in enumerate(candidate_ordering)}

    return bar_plot(
        data=score_dict,  # type: ignore[arg-type]
        data_set_label=profile_label,
        normalize=normalize,
        data_set_color=profile_color,
        bar_width=bar_width,
        category_ordering=candidate_ordering,
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name,
        title=title,
        show_data_set_legend=show_profile_legend,
        categories_legend=candidate_legend,
        threshold_values=threshold_values,
        threshold_kwds=threshold_kwds,
        legend_font_size=legend_font_size,
        ax=ax,
    )


def profile_mentions_plot(
    profile: PreferenceProfile,
    *,
    profile_label: str = "Profile",
    mentions_kwds: Optional[dict[str, Any]] = None,
    normalize: bool = False,
    profile_color: str = COLOR_LIST[0],
    bar_width: Optional[float] = None,
    candidate_ordering: Optional[list[str]] = None,
    x_axis_name: Optional[str] = None,
    y_axis_name: Optional[str] = None,
    title: Optional[str] = None,
    show_profile_legend: bool = False,
    candidate_legend: Optional[dict[str, str]] = None,
    relabel_candidates_with_int: bool = False,
    threshold_values: Optional[Union[list[float], float]] = None,
    threshold_kwds: Optional[Union[list[dict], dict]] = None,
    legend_font_size: Optional[float] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plots mentions of candidates in profile. Wrapper for ``profile_bar_plot``.

    Args:
        profile (PreferenceProfile): Profile to plot statistics for.
        profile_label (str, optional): Label for profile. Defaults to "Profile".
        mentions_kwds (dict[str, Any], optional): Keyword arguments to pass to
            ``mentions``. Defaults to None, in which case default values for ``mentions``
            are used.
        normalize (bool, optional): Whether or not to normalize data. Defaults to False.
        profile_color (str, optional): Color to plot. Defaults to the first color from
            ``COLOR_LIST`` from ``utils`` module.
        bar_width (float, optional): Width of bars. Defaults to None which computes the bar width
            as 0.7 divided by the number of data sets. Must be in the interval :math:`(0,1]`.
        candidate_ordering (list[str], optional): Ordering of x-labels. Defaults to decreasing
            order of mentions.
        x_axis_name (str, optional): Name of x-axis. Defaults to None, which does not plot a name.
        y_axis_name (str, optional): Name of y-axis. Defaults to None, which does not plot a name.
        title (str, optional): Title for the figure. Defaults to None, which does not plot a title.
        show_profile_legend (bool, optional): Whether or not to plot the profile legend.
            Defaults to False. Is automatically shown if any threshold lines have the keyword
            "label" passed through ``threshold_kwds``.
        candidate_legend (dict[str, str], optional): Dictionary mapping candidates
            to alternate label. Defaults to None. If provided, generates a second legend.
        relabel_candidates_with_int (bool, optional): Relabel the candidates with integer labels.
            Defaults to False. If ``candidate_legend`` is passed, those labels supercede.
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
    score_dict = (
        mentions(profile, **mentions_kwds) if mentions_kwds else mentions(profile)
    )

    if candidate_ordering is None:
        candidate_ordering = sorted(
            score_dict.keys(), reverse=True, key=lambda x: score_dict[x]
        )

    if relabel_candidates_with_int and not candidate_legend:
        candidate_legend = {c: str(i + 1) for i, c in enumerate(candidate_ordering)}

    return bar_plot(
        data=score_dict,  # type: ignore[arg-type]
        data_set_label=profile_label,
        normalize=normalize,
        data_set_color=profile_color,
        bar_width=bar_width,
        category_ordering=candidate_ordering,
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name,
        title=title,
        show_data_set_legend=show_profile_legend,
        categories_legend=candidate_legend,
        threshold_values=threshold_values,
        threshold_kwds=threshold_kwds,
        legend_font_size=legend_font_size,
        ax=ax,
    )


def profile_fpv_plot(
    profile: PreferenceProfile,
    *,
    profile_label: str = "Profile",
    fpv_kwds: Optional[dict[str, Any]] = None,
    normalize: bool = False,
    profile_color: str = COLOR_LIST[0],
    bar_width: Optional[float] = None,
    candidate_ordering: Optional[list[str]] = None,
    x_axis_name: Optional[str] = None,
    y_axis_name: Optional[str] = None,
    title: Optional[str] = None,
    show_profile_legend: bool = False,
    candidate_legend: Optional[dict[str, str]] = None,
    relabel_candidates_with_int: bool = False,
    threshold_values: Optional[Union[list[float], float]] = None,
    threshold_kwds: Optional[Union[list[dict], dict]] = None,
    legend_font_size: Optional[float] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plots first place votes of candidates in profile. Wrapper for ``profile_bar_plot``.

    Args:
        profile (PreferenceProfile): Profile to plot statistics for.
        profile_label (str, optional): Label for profile. Defaults to "Profile".
        fpv_kwds (dict[str, Any], optional): Keyword arguments to pass to
            ``first_place_votes``. Defaults to None, in which case default values for
            ``first_place_votes`` are used.
        normalize (bool, optional): Whether or not to normalize data. Defaults to False.
        profile_color (str, optional): Color to plot. Defaults to the first color from
            ``COLOR_LIST`` from ``utils`` module.
        bar_width (float, optional): Width of bars. Defaults to None which computes the bar width
            as 0.7 divided by the number of data sets. Must be in the interval :math:`(0,1]`.
        candidate_ordering (list[str], optional): Ordering of x-labels. Defaults to decreasing
            order of first place votes.
        x_axis_name (str, optional): Name of x-axis. Defaults to None, which does not plot a name.
        y_axis_name (str, optional): Name of y-axis. Defaults to None, which does not plot a name.
        title (str, optional): Title for the figure. Defaults to None, which does not plot a title.
        show_profile_legend (bool, optional): Whether or not to plot the profile legend.
            Defaults to False. Is automatically shown if any threshold lines have the keyword
            "label" passed through ``threshold_kwds``.
        candidate_legend (dict[str, str], optional): Dictionary mapping candidates
            to alternate label. Defaults to None. If provided, generates a second legend.
        relabel_candidates_with_int (bool, optional): Relabel the candidates with integer labels.
            Defaults to False. If ``candidate_legend`` is passed, those labels supercede.
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
    score_dict = (
        first_place_votes(profile, **fpv_kwds)
        if fpv_kwds
        else first_place_votes(profile)
    )

    if candidate_ordering is None:
        candidate_ordering = sorted(
            score_dict.keys(), reverse=True, key=lambda x: score_dict[x]
        )

    if relabel_candidates_with_int and not candidate_legend:
        candidate_legend = {c: str(i + 1) for i, c in enumerate(candidate_ordering)}

    return bar_plot(
        data=score_dict,  # type: ignore[arg-type]
        data_set_label=profile_label,
        normalize=normalize,
        data_set_color=profile_color,
        bar_width=bar_width,
        category_ordering=candidate_ordering,
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name,
        title=title,
        show_data_set_legend=show_profile_legend,
        categories_legend=candidate_legend,
        threshold_values=threshold_values,
        threshold_kwds=threshold_kwds,
        legend_font_size=legend_font_size,
        ax=ax,
    )


def profile_ballot_lengths_plot(
    profile: PreferenceProfile,
    *,
    profile_label: str = "Profile",
    ballot_lengths_kwds: Optional[dict[str, Any]] = None,
    normalize: bool = False,
    profile_color: str = COLOR_LIST[0],
    bar_width: Optional[float] = None,
    lengths_ordering: Optional[list[str]] = None,
    x_axis_name: Optional[str] = None,
    y_axis_name: Optional[str] = None,
    title: Optional[str] = None,
    show_profile_legend: bool = False,
    lengths_legend: Optional[dict[str, str]] = None,
    threshold_values: Optional[Union[list[float], float]] = None,
    threshold_kwds: Optional[Union[list[dict], dict]] = None,
    legend_font_size: Optional[float] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plots ballot lengths in profile. Wrapper for ``profile_bar_plot``.

    Args:
        profile (PreferenceProfile): Profile to plot statistics for.
        profile_label (str, optional): Label for profile. Defaults to "Profile".
        ballot_lengths_kwds (dict[str, Any], optional): Keyword arguments to pass to
            ``ballot_lengths``. Defaults to None, in which case default values for
            ``ballot_lengths`` are used.
        normalize (bool, optional): Whether or not to normalize data. Defaults to False.
        profile_color (str, optional): Color to plot. Defaults to the first color from
            ``COLOR_LIST`` from ``utils`` module.
        bar_width (float, optional): Width of bars. Defaults to None which computes the bar width
            as 0.7 divided by the number of data sets. Must be in the interval :math:`(0,1]`.
        lengths_ordering (list[str], optional): Ordering of x-labels. Defaults to increasing
            order of lengths.
        x_axis_name (str, optional): Name of x-axis. Defaults to None, which does not plot a name.
        y_axis_name (str, optional): Name of y-axis. Defaults to None, which does not plot a name.
        title (str, optional): Title for the figure. Defaults to None, which does not plot a title.
        show_profile_legend (bool, optional): Whether or not to plot the profile legend.
            Defaults to False. Is automatically shown if any threshold lines have the keyword
            "label" passed through ``threshold_kwds``.
        lengths_legend (dict[str, str], optional): Dictionary mapping lengths
            to alternate label. Defaults to None. If provided, generates a second legend.
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
        legend_bbox_to_anchor (Tuple[float, float], optional): The bounding box to anchor
            the legend to. Defaults to (1, 0.5).
        ax (Axes, optional): A matplotlib axes object to plot the figure on. Defaults to None, in
            which case the function creates and returns a new axes. The figure height is 6 inches
            and the figure width is 3 inches times the number of categories.

    Returns:
        Axes: A ``matplotlib`` axes with a bar plot of the given data.
    """
    score_dict = (
        ballot_lengths(profile, **ballot_lengths_kwds)
        if ballot_lengths_kwds
        else ballot_lengths(profile)
    )

    if lengths_ordering is None:
        lengths_ordering = sorted(
            score_dict.keys(), reverse=False, key=lambda x: x  # type: ignore[arg-type]
        )

    return bar_plot(
        data=score_dict,  # type: ignore[arg-type]
        data_set_label=profile_label,
        normalize=normalize,
        data_set_color=profile_color,
        bar_width=bar_width,
        category_ordering=lengths_ordering,
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name,
        title=title,
        show_data_set_legend=show_profile_legend,
        categories_legend=lengths_legend,
        threshold_values=threshold_values,
        threshold_kwds=threshold_kwds,
        legend_font_size=legend_font_size,
        ax=ax,
    )
