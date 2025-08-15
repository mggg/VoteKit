# tests/matrices/heatmap/test_heatmap_snapshots.py
import numpy as np
import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import LinearSegmentedColormap
from votekit.matrices.heatmap import matrix_heatmap  # adjust import if needed
from matplotlib.colors import ListedColormap


# Subtle custom sequential map for visibility of spacing/grid color
greys_steps = ListedColormap(
    ["#f9f9f9", "#dcdcdc", "#bfbfbf", "#a3a3a3", "#878787"], name="greys_steps"
)


def _fig_for_heatmap(M, **kwargs):
    fig, ax = plt.subplots()
    matrix_heatmap(M, ax=ax, **kwargs)
    fig.tight_layout()
    return fig


POSITIVE = np.array(
    [
        [0.00, 0.10, 0.30, 0.50],
        [0.20, 0.40, 0.60, 0.80],
        [0.10, 0.20, 0.70, 1.00],
    ]
)

DIVERGING = np.array(
    [
        [-1.2, -0.5, 0.0, 0.4, 1.1],
        [-0.9, -0.2, 0.0, 0.3, 0.8],
        [-0.6, -0.1, 0.0, 0.2, 0.5],
        [-0.8, -0.3, 0.0, 0.1, 0.7],
    ]
)

WITH_NAN_INF = np.array(
    [
        [np.nan, 0.2, 0.4],
        [np.inf, 0.1, 0.0],
    ]
)

WIDE_POS = np.array(
    [
        [0.00, 0.25, 0.50, 0.75, 1.00, 1.25],
        [0.10, 0.30, 0.55, 0.80, 0.95, 1.10],
    ]
)

TALL_NEG = np.array(
    [
        [-1.0, -0.5, -0.2],
        [-0.8, -0.3, -0.1],
        [-0.6, -0.2, -0.05],
        [-0.4, -0.1, -0.01],
    ]
)

mint_grape = LinearSegmentedColormap.from_list(
    "mint_grape", ["#1b9e77", "#f7f7f7", "#762a83"], N=256
)

CASES = [
    pytest.param(
        POSITIVE,
        dict(
            show_cell_values=True,
            n_decimals_to_display=2,
            row_labels=["r1", "r2", "r3"],
            column_labels=["c1", "c2", "c3", "c4"],
            show_colorbar=True,
            cell_spacing=0.5,
        ),
        id="heatmap_positive.png",
        marks=pytest.mark.mpl_image_compare(
            baseline_dir="tests/snapshots/heatmap",
            filename="heatmap_positive.png",
            tolerance=2,
        ),
    ),
    pytest.param(
        DIVERGING,
        dict(
            show_cell_values=False,
            show_colorbar=True,
            cell_spacing=0.5,
        ),
        id="heatmap_diverging.png",
        marks=pytest.mark.mpl_image_compare(
            baseline_dir="tests/snapshots/heatmap",
            filename="heatmap_diverging.png",
            tolerance=2,
        ),
    ),
    pytest.param(
        DIVERGING,
        dict(
            cell_color_map=mint_grape,
            row_labels=["A", "B", "C", "D"],
            column_labels=["W", "X", "Y", "Z", "Q"],
            row_label_rotation=0,
            column_label_rotation=45,
            show_colorbar=True,
            cell_spacing=0.3,
        ),
        id="heatmap_custom_cmap.png",
        marks=pytest.mark.mpl_image_compare(
            baseline_dir="tests/snapshots/heatmap",
            filename="heatmap_custom_cmap.png",
            tolerance=2,
        ),
    ),
    pytest.param(
        POSITIVE,
        dict(
            show_cell_values=False,
            show_colorbar=False,
            cell_spacing=1.0,
            cell_divider_color="white",
        ),
        id="heatmap_no_values_no_colorbar.png",
        marks=pytest.mark.mpl_image_compare(
            baseline_dir="tests/snapshots/heatmap",
            filename="heatmap_no_values_no_colorbar.png",
            tolerance=2,
        ),
    ),
    pytest.param(
        WITH_NAN_INF,
        dict(
            show_cell_values=True,
            n_decimals_to_display=1,
            column_labels=["c1", "c2", "c3"],
            row_labels=["r1", "r2"],
            show_colorbar=True,
            cell_spacing=0.5,
        ),
        id="heatmap_with_nan_inf.png",
        marks=pytest.mark.mpl_image_compare(
            baseline_dir="tests/snapshots/heatmap",
            filename="heatmap_with_nan_inf.png",
            tolerance=2,
        ),
    ),
    pytest.param(
        WIDE_POS,
        dict(
            show_cell_values=True,
            n_decimals_to_display=4,
            row_labels=["row $\\alpha$", "row $\\beta$"],
            column_labels=["c1", "c2", "c3", "c4", "c5", "c6"],
            row_label_rotation=30,
            column_label_rotation=75,
            show_colorbar=False,
            cell_spacing=0.5,
        ),
        id="heatmap_rotations_decimals.png",
        marks=pytest.mark.mpl_image_compare(
            baseline_dir="tests/snapshots/heatmap",
            filename="heatmap_rotations_decimals.png",
            tolerance=2,
        ),
    ),
    pytest.param(
        WIDE_POS,
        dict(
            show_cell_values=False,
            cell_spacing=1.5,
            cell_divider_color="black",
            show_colorbar=False,
        ),
        id="heatmap_heavy_grid.png",
        marks=pytest.mark.mpl_image_compare(
            baseline_dir="tests/snapshots/heatmap",
            filename="heatmap_heavy_grid.png",
            tolerance=2,
        ),
    ),
    pytest.param(
        WITH_NAN_INF,
        dict(
            cell_color_map=greys_steps,
            show_cell_values=True,
            n_decimals_to_display=1,
            row_labels=["r1", "r2"],
            column_labels=["x", "y", "z"],
            show_colorbar=True,
            cell_spacing=0.6,
            cell_divider_color="#444444",
        ),
        id="heatmap_custom_seq_nan_inf.png",
        marks=pytest.mark.mpl_image_compare(
            baseline_dir="tests/snapshots/heatmap",
            filename="heatmap_custom_seq_nan_inf.png",
            tolerance=2,
        ),
    ),
    pytest.param(
        TALL_NEG,
        dict(
            cell_color_map=mint_grape,
            row_labels=["A", "B", "C", "D"],
            column_labels=["u", "v", "w"],
            show_cell_values=True,
            n_decimals_to_display=4,
            show_colorbar=True,
            cell_spacing=0.4,
        ),
        id="heatmap_all_negative_custom_map.png",
        marks=pytest.mark.mpl_image_compare(
            baseline_dir="tests/snapshots/heatmap",
            filename="heatmap_all_negative_custom_map.png",
            tolerance=2,
        ),
    ),
    pytest.param(
        POSITIVE,
        dict(
            row_labels=["r1", "r2", "r3"],
            column_labels=["c1", "c2", "c3", "c4"],
            row_legend={
                "r1": "Row 1: seed",
                "r2": "Row 2: control",
                "r3": "Row 3: treated",
            },
            column_legend={
                "c1": "Stage 1",
                "c2": "Stage 2",
                "c3": "Stage 3",
                "c4": "Stage 4",
            },
            legend_font_size=9.5,
            legend_location="center left",
            legend_bbox_to_anchor=(1.02, 0.5),
            show_cell_values=False,
            show_colorbar=False,
            cell_spacing=0.5,
        ),
        id="heatmap_legends_location_font.png",
        marks=pytest.mark.mpl_image_compare(
            baseline_dir="tests/snapshots/heatmap",
            filename="heatmap_legends_location_right.png",
            tolerance=3,
        ),
    ),
    pytest.param(
        POSITIVE,
        dict(
            show_cell_values=True,
            cell_font_size=6,
            row_labels=["r1", "r2", "r3"],
            column_labels=["c1", "c2", "c3", "c4"],
            show_colorbar=False,
            cell_spacing=0.3,
        ),
        id="heatmap_small_cell_font.png",
        marks=pytest.mark.mpl_image_compare(
            baseline_dir="tests/snapshots/heatmap",
            filename="heatmap_small_cell_font.png",
            tolerance=2,
        ),
    ),
]


@pytest.mark.parametrize("M,kwargs", CASES)
def test_matrix_heatmap_snapshots(M, kwargs):
    fig = _fig_for_heatmap(M, **kwargs)
    return fig
