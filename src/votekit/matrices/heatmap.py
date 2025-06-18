import numpy as np
from matplotlib.axes import Axes
from typing import Optional, Dict, List, Tuple, Union
from matplotlib.colors import Colormap, TwoSlopeNorm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches


def _validate_heatmap_inputs(
    *,
    matrix: np.ndarray,
    row_labels: Optional[List[str]] = None,
    column_labels: Optional[List[str]] = None,
    row_legend: Optional[Dict[str, str]] = None,
    column_legend: Optional[Dict[str, str]] = None,
) -> None:
    """
    Used internally for the `matrix_heatmap` function to validate the inputs to the function.

    Args:
        matrix (np.ndarray): A 2D numpy array containing the data to be plotted.
        row_labels (Optional[List[str]]): A list of strings containing the labels for the rows
            of the heatmap. Defaults to None.
        column_labels (Optional[List[str]]): A list of strings containing the labels for the
            columns of the heatmap. Defaults to None.
        row_legend (Optional[Dict[str, str]]): A dictionary mapping row labels to legend
            descriptions. Defaults to None.
        column_legend (Optional[Dict[str, str]]): A dictionary mapping column labels to legend
            descriptions. Defaults to None.

    Raises:
        ValueError: If the matrix is not a 2D numpy array.
        ValueError: If the row labels are not of the correct length.
        ValueError: If the column labels are not of the correct length.
        ValueError: If the keys of the row legend are do not match the row labels.
        ValueError: If the keys of the column legend are do not match the column labels.
        ValueError: If the legend descriptions in the row legend and column legend are in conflict.
    """
    if matrix.ndim != 2:
        raise ValueError(
            f"Please provide a 2D matrix to plot. Found a {matrix.ndim}-D matrix."
        )

    n_rows = matrix.shape[0]
    n_cols = matrix.shape[1]

    if row_labels is not None:
        if n_rows != len(row_labels):
            raise ValueError(
                f"Please provide {n_rows} labels for the rows of the "
                f"matrix. Found {len(row_labels)} labels."
            )

    if column_labels is not None:
        if n_cols != len(column_labels):
            raise ValueError(
                f"Please provide {n_cols} labels for the columns of the "
                f"matrix. Found {len(column_labels)} labels."
            )

    if row_legend is not None and column_legend is not None:
        for label, description in column_legend.items():
            desc = row_legend.get(label, description)
            if desc != description:
                raise ValueError(
                    f"Conflicting legend descriptions for '{label}': "
                    f"got '{description}' and '{desc}'."
                )

    if row_legend is not None and list(row_legend.keys()) != row_labels:
        raise ValueError("Row labels do not match row legend keys.")

    if column_legend is not None and list(column_legend.keys()) != column_labels:
        raise ValueError("Column labels do not match column legend keys.")


def _add_text_to_heatmap(
    *,
    heatmap: Axes,
    n_decimals_to_display: int,
    matrix: np.ndarray,
    cell_font_size: Optional[int] = None,
) -> Axes:
    """
    Adds the text values to the heatmap cells. This function dynamically determines
    the font size based on the number of cells in the figure, the figure size, and
    the length of the numbers to ensure readability.

    Args:
        heatmap (matplotlib.axes.Axes): The matplotlib axis containing the heatmap.
        n_decimals_to_display (int): The number of decimal places to display for the values
            in the heatmap.
        matrix (np.ndarray): A 2D numpy array containing the data to be plotted.
        cell_font_size (Optional[int]): The base font size to use for the cell values.
            If None, the font size will be dynamically determined.

    Returns:
        matplotlib.axes.Axes: The matplotlib axis containing the heatmap with the cell values
            and text values added.
    """
    nrows, ncols = matrix.shape
    quadmesh = heatmap.collections[0]

    max_chars = max(
        len(f"{val:.{n_decimals_to_display}f}") + (3 if val < 0 else 2)
        for val in matrix.flatten()
    )

    if cell_font_size is not None:
        font_size = cell_font_size
    else:
        fig = heatmap.get_figure()
        if fig:
            fig.canvas.draw()

            # ignoring mypy error, mypy not up to date with matplotlib get_renderer
            renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]

            # bounding box in display coords
            bbox = heatmap.get_window_extent(renderer=renderer)
            width_pts = bbox.width

            cell_width_pts = width_pts / ncols

            font_size = int(cell_width_pts / max_chars)

    for i in range(nrows):
        for j in range(ncols):
            val = matrix[i, j]
            txt = "N/A" if np.isnan(val) else f"{val:.{n_decimals_to_display}f}"

            # Normalize the cell value between 0 and 1, then get the RGBA color
            norm_val = quadmesh.norm(val)
            r, g, b, _ = quadmesh.cmap(norm_val)  # type: ignore[misc]

            # Simple brightness measure: average of R, G, B
            brightness = (r + g + b) / 3
            txt_color = "black" if brightness > 0.5 else "white"

            heatmap.text(
                j + 0.5,
                i + 0.5,
                txt,
                ha="center",
                va="center",
                color=txt_color,
                fontsize=font_size,
            )

    return heatmap


def _add_legend_to_heatmap(
    *,
    row_and_col_legends: Dict[str, str],
    ax: Axes,
    legend_font_size: float,
    legend_loc: str,
    legend_bbox_to_anchor: Tuple[float, float],
) -> Axes:
    """
    Adds a legend to a heatmap.

    Args:
        row_and_col_legends (Dict[str, str]): A dictionary mapping row and column labels to
            legend descriptions.
        ax (matplotlib.axes.Axes): The matplotlib axis to add the legend to.
        legend_font_size (float): The font size to use for the legend.
        legend_loc (str): The location to place the legend.
        legend_bbox_to_anchor (Tuple[float, float]): The bounding box to anchor the legend to.

    Returns:
        matplotlib.axes.Axes: The matplotlib axis containing with the updated legend.
    """
    proxy_artists = []
    proxy_labels = []

    if len(row_and_col_legends) != 0:
        for label, description in row_and_col_legends.items():
            patch = mpatches.Patch(color="white", label=f"{label}: {description}")
            proxy_artists.append(patch)
            proxy_labels.append(f"{label}: {description}")

    if proxy_artists:
        leg = ax.legend(
            handles=proxy_artists,
            labels=proxy_labels,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            fontsize=legend_font_size,
            ncol=len(proxy_labels) // 15 + 1,
            frameon=True,
            borderaxespad=0.0,
            handlelength=0,
            handletextpad=0,
            fancybox=True,
        )

        for item in leg.legend_handles:
            if item:
                item.set_visible(False)

    return ax


def matrix_heatmap(
    matrix: np.ndarray,
    *,
    ax: Optional[Axes] = None,
    show_cell_values: bool = True,
    n_decimals_to_display: int = 2,
    row_labels: Optional[List[str]] = None,
    row_label_rotation: Optional[float] = None,
    row_legend: Optional[dict[str, str]] = None,
    column_labels: Optional[List[str]] = None,
    column_label_rotation: Optional[float] = None,
    column_legend: Optional[dict[str, str]] = None,
    cell_color_map: Optional[Union[str, Colormap]] = None,
    cell_font_size: Optional[int] = None,
    cell_spacing: float = 0.5,
    cell_divider_color: str = "white",
    show_colorbar: bool = False,
    legend_font_size: float = 10.0,
    legend_location: str = "center left",
    legend_bbox_to_anchor: Tuple[float, float] = (1.03, 0.5),
) -> Axes:
    """
    Basic function for plotting a matrix as a heatmap.

    Args:
        matrix (np.ndarray): A 2D numpy array containing the data to be plotted.
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. Defaults to None,
            in which case an axis is created.
        show_cell_values (bool): Whether to show the values of the cells in the heatmap. These
            values are shown in the center of each cell and are dynamically formatted to be
            human-readable.  Defaults to True.
        n_decimals_to_display (int): The number of decimal places to display for the values
            in the heatmap.  Defaults to 2.
        row_labels (Optional(List[str])): A list of strings containing the labels for the rows
            of the heatmap. Defaults to None.
        row_label_rotation (Optional(float)): The rotation to apply to the row labels.
            Defaults to None.
        row_legend (Optional(Dict[str, str])): A dictionary mapping row labels to legend
            descriptions. Defaults to None.
        column_labels (Optional(List[str])): A list of strings containing the labels for the
            columns of the heatmap. Defaults to None.
        column_label_rotation (Optional(float)): The rotation to apply to the column labels.
            Defaults to None.
        column_legend (Optional(Dict[str, str])): A dictionary mapping column labels to legend
            descriptions. Defaults to None.
        cell_color_map (Optional(Union[str, matplotlib.colors.Colormap])): The color map to use
            for the heatmap. Defaults to `PRGn` if the matrix contains negative values and
            `Greens` otherwise.
        cell_font_size (Optional(int)): The font size to use for the cell values. Defaults to
            None, which will then use dynamic font size based on the number of cells and the
            figure size.
        cell_spacing (float): The spacing between the cells in the heatmap. Defaults to 0.5.
        cell_divider_color (str): The color to use for the cell dividers for spacing cells.
            Defaults to "white".
        show_colorbar (bool): Whether to show the colorbar for the heatmap. Defaults to False.
        legend_font_size (float): The font size to use for the legend. Defaults to 10.0.
        legend_location (str): The location to place the legend. Defaults to "center left".
        legend_bbox_to_anchor (Tuple[float, float]): The bounding box to anchor the legend to.
            Defaults to (1.03, 0.5).

    Returns:
        matplotlib.axes.Axes: The matplotlib axis containing the heatmap.
    """

    if ax is None:
        fig, ax = plt.subplots()

    _validate_heatmap_inputs(
        matrix=matrix,
        row_labels=row_labels,
        column_labels=column_labels,
        row_legend=row_legend,
        column_legend=column_legend,
    )

    row_and_col_legends = dict()
    if row_legend is not None:
        row_and_col_legends.update(row_legend)
    if column_legend is not None:
        row_and_col_legends.update(column_legend)

    ax.xaxis.set_ticks_position("top")

    if cell_color_map is None:
        if np.nanmin(matrix) < 0:
            cell_color_map = sns.color_palette("PRGn", as_cmap=True)
            norm: Union[TwoSlopeNorm, str] = TwoSlopeNorm(
                vmin=np.nanmin(matrix), vcenter=0.0, vmax=np.nanmax(matrix)
            )
        else:
            cell_color_map = sns.color_palette("Greens", as_cmap=True)
            norm = "linear"

    heatmap = sns.heatmap(
        matrix,
        ax=ax,
        cmap=cell_color_map,
        norm=norm,
        fmt=f".{n_decimals_to_display}f",
        linewidths=cell_spacing,
        linecolor=cell_divider_color,
        cbar=show_colorbar,
        yticklabels=row_labels if row_labels is not None else False,
        xticklabels=column_labels if column_labels is not None else False,
    )

    plt.gca().set_facecolor("black")  # inf cells

    if show_cell_values:
        heatmap = _add_text_to_heatmap(
            heatmap=heatmap,
            n_decimals_to_display=n_decimals_to_display,
            matrix=matrix,
            cell_font_size=cell_font_size,
        )

    if len(row_and_col_legends) > 0:
        heatmap = _add_legend_to_heatmap(
            row_and_col_legends=row_and_col_legends,
            ax=heatmap,
            legend_font_size=legend_font_size,
            legend_loc=legend_location,
            legend_bbox_to_anchor=legend_bbox_to_anchor,
        )

    if column_label_rotation is not None:
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=column_label_rotation,
            ha="left",
            rotation_mode="anchor",
        )
    if row_label_rotation is not None:
        ax.set_yticklabels(
            ax.get_yticklabels(), rotation=row_label_rotation, rotation_mode="anchor"
        )

    return ax
