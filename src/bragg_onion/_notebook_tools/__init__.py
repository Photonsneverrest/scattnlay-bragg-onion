"""
Notebook helper utilities for analysis / plotting workflows.

These helpers are intentionally kept lightweight and should not import the
solver/scattnlay stack.
"""

from .io import (
    running_in_wsl,
    windows_to_wsl_path,
    detect_parquet_support,
    load_sweep_outputs,
)

from .colour import (
    add_display_normalized_colour_columns,
)

from .strip_plots import (
    select_best_etaC_by_wavelength,
    plot_best_etaC_colour_strip,
    select_strip_grid,
    plot_colour_strip_grid,
    plot_sweep_heatmap_local,
)

from .rosch_plots import (
    make_rosch_macadam_figures,
    add_highlight_cielab_points,
)

__all__ = [
    "running_in_wsl",
    "windows_to_wsl_path",
    "detect_parquet_support",
    "load_sweep_outputs",
    "add_display_normalized_colour_columns",
    "select_best_etaC_by_wavelength",
    "plot_best_etaC_colour_strip",
    "select_strip_grid",
    "plot_colour_strip_grid",
    "plot_sweep_heatmap_local",
    "make_rosch_macadam_figures",
    "add_highlight_cielab_points",
]