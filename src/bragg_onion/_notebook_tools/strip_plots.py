from __future__ import annotations

from typing import Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
from collections.abc import Callable


def _apply_fixed_filters(df: pd.DataFrame, fixed_filters: dict | None) -> pd.DataFrame:
    """
    Apply fixed filters to a dataframe.
    Uses np.isclose for numeric scalar values.
    """
    if fixed_filters is None:
        return df.copy()

    out = df.copy()
    for col, val in fixed_filters.items():
        if col not in out.columns:
            raise KeyError(f"Fixed-filter column '{col}' is not in dataframe.")

        if np.issubdtype(out[col].dtype, np.number) and isinstance(val, (int, float, np.number)):
            out = out[np.isclose(out[col].astype(float), float(val))]
        else:
            out = out[out[col] == val]

    return out


def _text_color_for_hex(hex_color: str) -> str:
    """
    Choose black or white annotation text depending on tile brightness.
    """
    try:
        r, g, b = to_rgb(hex_color)
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "black" if luminance > 0.55 else "white"
    except Exception:
        return "black"


def _pretty_metric_label(col: str) -> str:
    """
    Pretty display label for known sweep metrics.
    """
    mapping = {
        "eta_C": "η_C",
        "eta_L": "η_L",
        "eta_Y": "η_Y",
    }
    return mapping.get(col, col)


def _format_annotation(row, annotate_cols, annotation_fmt=".2f"):
    """
    Format tile annotation from one or more columns.

    Special handling:
      - eta_C, eta_L, eta_Y are shown as percentages with no decimals
      - other numeric values use annotation_fmt
    """
    if annotate_cols is None:
        return ""

    if isinstance(annotate_cols, str):
        annotate_cols = [annotate_cols]

    lines = []
    for col in annotate_cols:
        if col not in row.index:
            continue

        val = row[col]
        label = _pretty_metric_label(col)

        try:
            if isinstance(val, (float, np.floating, int, np.integer)):
                if col in {"eta_C", "eta_L", "eta_Y"}:
                    text = format(float(val), ".0%")
                else:
                    text = format(val, annotation_fmt)
            else:
                text = str(val)
        except Exception:
            text = str(val)

        lines.append(f"{label}={text}")

    return "\n".join(lines)

# ------------------------------------------------------------
# Public formatting / preparation helpers
# ------------------------------------------------------------
def pretty_param_name(name: str) -> str:
    """
    Pretty display label for common sweep parameters and metrics.
    """
    mapping = {
        "n_layers": "n_layers",
        "n_A": "n_A",
        "k": "k",
        "collection_angle_deg": "collection angle",
        "design_peak_wavelength_nm": "design wavelength",
        "outer_layer": "outer layer",
        "case_index": "case",
        "eta_C": "η_C",
        "eta_L": "η_L",
        "eta_Y": "η_Y",
    }
    return mapping.get(name, name)


def format_param_value(name: str, value) -> str:
    """
    Pretty formatting for parameter values.
    """
    if isinstance(value, (float, np.floating)):
        if name == "collection_angle_deg":
            return f"{value:.1f}°"
        if name in {"k", "n_A"}:
            return f"{value:.3f}"
        if float(value).is_integer():
            return f"{int(value)}"
        return f"{value:.3f}"
    return str(value)


def make_row_label_fmt(row_variable: str) -> Callable:
    """
    Return a formatter function suitable for plot_colour_strip_grid(..., row_label_fmt=...)
    """
    def _fmt(v):
        return f"{row_variable}={format_param_value(row_variable, v)}"
    return _fmt


def make_strip_grid_title(row_variable: str, fixed_filters: dict) -> str:
    """
    Build a compact automatic title for the strip grid.
    """
    fixed_desc = ", ".join(
        f"{pretty_param_name(k)} = {format_param_value(k, v)}"
        for k, v in fixed_filters.items()
    )
    return (
        f"Colour-strip grid: rows = {row_variable}, columns = design wavelength\n"
        f"({fixed_desc})"
    )


def choose_colour_source_dataframe(
    df_all: pd.DataFrame,
    df_plot: pd.DataFrame,
    colour_hex_col: str,
) -> pd.DataFrame:
    """
    Automatically choose the correct dataframe based on the requested colour column.
    """
    return df_plot if colour_hex_col.endswith("_display") else df_all


def prepare_strip_grid(
    df_all: pd.DataFrame,
    df_plot: pd.DataFrame,
    *,
    row_variable: str,
    fixed_filters: dict,
    colour_hex_col: str,
    x: str = "design_peak_wavelength_nm",
    score_col: str = "eta_C",
    select_best_within_cell: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    High-level helper to prepare a strip-grid dataframe and all notebook-facing metadata.

    Returns
    -------
    df_grid : pd.DataFrame
        One row per plotted tile.
    meta : dict
        Contains:
          - fixed_filters_clean
          - row_label_fmt
          - title
          - row_variable
          - colour_hex_col
          - df_source_name
    """
    fixed_filters_clean = fixed_filters.copy()

    removed_filter = None
    if row_variable in fixed_filters_clean:
        removed_filter = fixed_filters_clean.pop(row_variable)

    df_source = choose_colour_source_dataframe(df_all, df_plot, colour_hex_col)
    df_source_name = "df_plot" if df_source is df_plot else "df_all"

    df_grid = select_strip_grid(
        df_source,
        row_variable=row_variable,
        fixed_filters=fixed_filters_clean,
        x=x,
        score_col=score_col,
        select_best_within_cell=select_best_within_cell,
    )

    meta = {
        "fixed_filters_clean": fixed_filters_clean,
        "row_label_fmt": make_row_label_fmt(row_variable),
        "title": make_strip_grid_title(row_variable, fixed_filters_clean),
        "row_variable": row_variable,
        "colour_hex_col": colour_hex_col,
        "df_source_name": df_source_name,
        "removed_filter": removed_filter,
        "x": x,
        "score_col": score_col,
    }

    return df_grid, meta


def summarize_strip_grid_cases(
    df_grid: pd.DataFrame,
    *,
    row_variable: str,
    fixed_filters: dict,
    colour_hex_col: str,
    x: str = "design_peak_wavelength_nm",
) -> pd.DataFrame:
    """
    Return a compact summary table of the actually plotted strip-grid tiles.

    Only includes:
      - x
      - case_index
      - the row variable
      - fixed filter parameters
      - η_C / η_L as percentages
      - selected colour column

    This keeps the printed table focused on the plotted grid rather than dumping many columns.
    """
    df_print = df_grid.copy()

    if "eta_C" in df_print.columns:
        df_print["eta_C_pct"] = (100 * df_print["eta_C"]).round(0).astype(int)
    if "eta_L" in df_print.columns:
        df_print["eta_L_pct"] = (100 * df_print["eta_L"]).round(0).astype(int)

    cols = [x, "case_index", row_variable]

    # only include the fixed parameters actually used for this plot
    for col in fixed_filters.keys():
        if col != row_variable:
            cols.append(col)

    cols.extend(["eta_C_pct", "eta_L_pct", colour_hex_col])

    # remove duplicates while preserving order
    cols_unique = []
    for c in cols:
        if c not in cols_unique:
            cols_unique.append(c)

    cols_unique = [c for c in cols_unique if c in df_print.columns]

    sort_cols = [c for c in [row_variable, x] if c in df_print.columns]
    df_print = df_print.sort_values(sort_cols).reset_index(drop=True)

    return df_print[cols_unique]

# ------------------------------------------------------------
# OPTION 1 — best eta_C single-strip mode
# ------------------------------------------------------------
def select_best_etaC_by_wavelength(
    df: pd.DataFrame,
    *,
    fixed_filters: dict | None = None,
    x: str = "design_peak_wavelength_nm",
    score_col: str = "eta_C",
) -> pd.DataFrame:
    """
    Select the best case (highest score_col) for each x value after applying optional fixed filters.
    Returns one row per x value.
    """
    if x not in df.columns:
        raise KeyError(f"x='{x}' is not a column in dataframe.")
    if score_col not in df.columns:
        raise KeyError(f"score_col='{score_col}' is not a column in dataframe.")

    df_sel = _apply_fixed_filters(df, fixed_filters)
    if df_sel.empty:
        raise ValueError("No rows remain after applying fixed filters.")

    df_best = (
        df_sel.sort_values(score_col, ascending=False)
              .groupby(x, as_index=False)
              .first()
              .sort_values(x)
              .reset_index(drop=True)
    )
    return df_best


def plot_best_etaC_colour_strip(
    df_best: pd.DataFrame,
    *,
    x: str = "design_peak_wavelength_nm",
    colour_hex_col: str = "hex",
    annotate_cols: str | list[str] | None = None,
    annotation_fmt: str = ".2f",
    title: str | None = None,
    figsize=(10, 1.8),
    ax=None,
):
    """
    Plot a single-row colour strip from a dataframe that already contains
    one best row per x value.
    """
    if x not in df_best.columns:
        raise KeyError(f"x='{x}' is not a column in dataframe.")
    if colour_hex_col not in df_best.columns:
        msg = f"colour_hex_col='{colour_hex_col}' is not a column in dataframe."
        if colour_hex_col.endswith("_display"):
            msg += (
                " If you want display-normalized colours, make sure you ran the "
                "display-normalized colour step and built df_best from the augmented dataframe."
            )
        raise KeyError(msg)

    data = df_best.sort_values(x).reset_index(drop=True)
    x_vals = data[x].to_numpy(dtype=float)

    if len(x_vals) == 0:
        raise ValueError("df_best is empty.")

    if len(x_vals) == 1:
        widths = np.array([1.0])
    else:
        dx = np.diff(x_vals)
        widths = np.empty_like(x_vals)
        widths[1:-1] = 0.5 * (dx[:-1] + dx[1:])
        widths[0] = dx[0]
        widths[-1] = dx[-1]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for i, row in data.iterrows():
        xv = float(row[x])
        w = widths[i]
        x0 = xv - 0.5 * w
        color = row[colour_hex_col]

        rect = Rectangle((x0, 0), w, 1, facecolor=color, edgecolor="black", linewidth=0.7)
        ax.add_patch(rect)

        annot = _format_annotation(row, annotate_cols, annotation_fmt)
        if annot:
            ax.text(
                xv,
                0.5,
                annot,
                ha="center",
                va="center",
                fontsize=8,
                color=_text_color_for_hex(color),
            )

    ax.set_xlim(x_vals.min() - widths[0], x_vals.max() + widths[-1])
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(x)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([f"{v:.0f}" for v in x_vals])

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    return ax


# ------------------------------------------------------------
# OPTION 2 — multi-row strip grid
# ------------------------------------------------------------
def select_strip_grid(
    df: pd.DataFrame,
    *,
    row_variable: str,
    fixed_filters: dict | None = None,
    x: str = "design_peak_wavelength_nm",
    score_col: str = "eta_C",
    select_best_within_cell: bool = True,
) -> pd.DataFrame:
    """
    Prepare a dataframe for multi-row strip grid plotting.

    If select_best_within_cell=True:
        For each (row_variable, x) combination, select the row with highest score_col.
    If False:
        requires exactly one row per (row_variable, x) after filtering.
    """
    if row_variable not in df.columns:
        raise KeyError(f"row_variable='{row_variable}' is not a column in dataframe.")
    if x not in df.columns:
        raise KeyError(f"x='{x}' is not a column in dataframe.")
    if score_col not in df.columns:
        raise KeyError(f"score_col='{score_col}' is not a column in dataframe.")

    df_sel = _apply_fixed_filters(df, fixed_filters)
    if df_sel.empty:
        raise ValueError("No rows remain after applying fixed filters.")

    if select_best_within_cell:
        df_grid = (
            df_sel.sort_values(score_col, ascending=False)
                  .groupby([row_variable, x], as_index=False)
                  .first()
                  .sort_values([row_variable, x])
                  .reset_index(drop=True)
        )
    else:
        counts = df_sel.groupby([row_variable, x]).size()
        if (counts > 1).any():
            raise ValueError(
                "Multiple rows per (row_variable, x) remain after filtering. "
                "Either fix more parameters or use select_best_within_cell=True."
            )
        df_grid = df_sel.sort_values([row_variable, x]).reset_index(drop=True)

    return df_grid


def plot_colour_strip_grid(
    df_grid: pd.DataFrame,
    *,
    row_variable: str,
    x: str = "design_peak_wavelength_nm",
    colour_hex_col: str = "hex",
    annotate_cols: str | list[str] | None = None,
    annotation_fmt: str = ".2f",
    title: str | None = None,
    row_label_fmt=None,
    figsize=None,
    overlay_line_cols: list[str] | None = None,
    overlay_line_colors: dict | None = None,
    overlay_line_style: dict | None = None,
    overlay_line_scale: Literal["row_separate", "row_shared", "global_separate", "global_shared"] = "global_shared",
    overlay_reference_values: list[float] | None = [0.0, 1.0],
    overlay_reference_color: str = "gray",
    overlay_reference_alpha: float = 0.55,
    overlay_reference_linestyle: str = "--",
):
    """
    Plot a multi-row colour strip grid:
      - columns: x values
      - rows: row_variable values
      - tile colour: colour_hex_col
      - optional annotations: annotate_cols
      - optional overlay line graphs within each row: overlay_line_cols
    """
    if row_variable not in df_grid.columns:
        raise KeyError(f"row_variable='{row_variable}' is not a column in dataframe.")
    if x not in df_grid.columns:
        raise KeyError(f"x='{x}' is not a column in dataframe.")
    if colour_hex_col not in df_grid.columns:
        msg = f"colour_hex_col='{colour_hex_col}' is not a column in dataframe."
        if colour_hex_col.endswith("_display"):
            msg += (
                " If you want display-normalized colours, make sure you ran the "
                "display-normalized colour step and built df_grid from the augmented dataframe."
            )
        raise KeyError(msg)

    if overlay_line_cols is None:
        overlay_line_cols = []

    if overlay_line_colors is None:
        overlay_line_colors = {
            "eta_C": "black",
            "eta_L": "white",
        }

    if overlay_line_style is None:
        overlay_line_style = {
            "linewidth": 1.5,
            "alpha": 0.95,
        }

    row_vals = list(df_grid[row_variable].dropna().unique())
    try:
        row_vals = sorted(row_vals)
    except Exception:
        row_vals = list(row_vals)

    x_vals = list(df_grid[x].dropna().unique())
    try:
        x_vals = sorted(x_vals)
    except Exception:
        x_vals = list(x_vals)

    n_rows = len(row_vals)
    n_cols = len(x_vals)

    if figsize is None:
        figsize = (0.85 * n_cols + 3.0, 0.75 * n_rows + 2.0)

    fig, ax = plt.subplots(figsize=figsize)

    global_minmax_per_col: dict[str, tuple[float, float]] = {}
    global_shared_minmax: Optional[tuple[float, float]] = None

    if overlay_line_cols:
        for col in overlay_line_cols:
            if col not in df_grid.columns:
                raise KeyError(f"overlay_line_col '{col}' is not in dataframe.")

        if overlay_line_scale == "global_separate":
            for col in overlay_line_cols:
                vals = df_grid[col].to_numpy(dtype=float)
                global_minmax_per_col[col] = (float(np.nanmin(vals)), float(np.nanmax(vals)))

        elif overlay_line_scale == "global_shared":
            all_vals = np.concatenate([
                df_grid[col].to_numpy(dtype=float)
                for col in overlay_line_cols
            ])
            global_shared_minmax = (float(np.nanmin(all_vals)), float(np.nanmax(all_vals)))

    def _normalize_values(vals, vmin: float, vmax: float):
        vals = np.asarray(vals, dtype=float)
        if np.isclose(vmax, vmin):
            return np.full_like(vals, 0.5, dtype=float)
        return (vals - vmin) / (vmax - vmin)

    def _get_scale_for_row(row_subset: pd.DataFrame, col: str) -> tuple[float, float]:
        if overlay_line_scale == "row_separate":
            vals = row_subset[col].to_numpy(dtype=float)
            return float(np.nanmin(vals)), float(np.nanmax(vals))

        elif overlay_line_scale == "row_shared":
            all_vals = np.concatenate([row_subset[c].to_numpy(dtype=float) for c in overlay_line_cols])
            return float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

        elif overlay_line_scale == "global_separate":
            if col not in global_minmax_per_col:
                raise RuntimeError(f"Global min/max for column '{col}' was not initialized.")
            return global_minmax_per_col[col]

        elif overlay_line_scale == "global_shared":
            if global_shared_minmax is None:
                raise RuntimeError("global_shared_minmax was not initialized.")
            return global_shared_minmax

        raise RuntimeError(f"Unexpected overlay_line_scale='{overlay_line_scale}'.")

    x_to_j = {xv: j for j, xv in enumerate(x_vals)}

    for i, row_val in enumerate(row_vals):
        y = n_rows - 1 - i  # top row first
        row_subset = df_grid[df_grid[row_variable] == row_val].copy().sort_values(x)

        # draw tiles
        for j, x_val in enumerate(x_vals):
            subset = row_subset[row_subset[x] == x_val]

            if subset.empty:
                color = "#ffffff"
                annot = ""
            else:
                row = subset.iloc[0]
                color = row[colour_hex_col]
                annot = _format_annotation(row, annotate_cols, annotation_fmt)

            rect = Rectangle((j, y), 1, 1, facecolor=color, edgecolor="black", linewidth=0.7)
            ax.add_patch(rect)

            if annot:
                ax.text(
                    j + 0.5,
                    y + 0.5,
                    annot,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=_text_color_for_hex(color),
                )

        # overlay reference lines
        if overlay_line_cols and overlay_reference_values:
            ref_col = overlay_line_cols[0]
            vmin_ref, vmax_ref = _get_scale_for_row(row_subset, ref_col)
            lo = min(vmin_ref, vmax_ref)
            hi = max(vmin_ref, vmax_ref)

            for ref_val in overlay_reference_values:
                if lo <= ref_val <= hi:
                    ref_norm = float(_normalize_values([ref_val], vmin_ref, vmax_ref)[0])
                    y_ref = y + 0.15 + 0.70 * ref_norm
                    ax.plot(
                        [0, n_cols],
                        [y_ref, y_ref],
                        color=overlay_reference_color,
                        alpha=overlay_reference_alpha,
                        linestyle=overlay_reference_linestyle,
                        linewidth=1.0,
                    )

        # overlay lines
        if overlay_line_cols:
            for col in overlay_line_cols:
                vals = row_subset[col].to_numpy(dtype=float)
                vmin, vmax = _get_scale_for_row(row_subset, col)
                norm_vals = _normalize_values(vals, vmin, vmax)

                xcenters = []
                yvals = []
                for xv, nv in zip(row_subset[x].to_numpy(dtype=float), norm_vals):
                    j = x_to_j[xv]
                    xcenters.append(j + 0.5)
                    yvals.append(y + 0.15 + 0.70 * float(nv))

                ax.plot(
                    xcenters,
                    yvals,
                    color=overlay_line_colors.get(col, "black"),
                    label=_pretty_metric_label(col) if i == 0 else None,
                    **overlay_line_style,
                )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(
        [f"{v:.0f}" if isinstance(v, (int, float, np.number)) else str(v) for v in x_vals],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel(x)

    ax.set_yticks(np.arange(n_rows) + 0.5)
    if row_label_fmt is None:
        yticklabels = [
            f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
            for v in row_vals
        ]
    else:
        yticklabels = [row_label_fmt(v) for v in row_vals]
    ax.set_yticklabels(yticklabels[::-1])  # because top row first
    ax.set_ylabel(row_variable)

    if title is not None:
        ax.set_title(title)

    if overlay_line_cols:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    ax.set_aspect("equal")
    plt.tight_layout()
    return fig, ax


def plot_sweep_heatmap_local(
    df: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    *,
    aggfunc: Literal["mean", "sum", "median", "min", "max"] = "mean",
    cmap: str = "viridis",
    ax=None,
):
    """
    Simple heatmap from a sweep dataframe.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.8))

    pivot = pd.pivot_table(
        df,
        index=y,
        columns=x,
        values=value,
        aggfunc=aggfunc,
    )

    pivot = pivot.sort_index().sort_index(axis=1)

    im = ax.imshow(
        pivot.to_numpy(),
        origin="lower",
        aspect="auto",
        cmap=cmap,
    )

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(
        [f"{c:.0f}" if isinstance(c, (int, float, np.number)) else str(c) for c in pivot.columns],
        rotation=45,
        ha="right",
    )

    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(
        [f"{r:.0f}" if isinstance(r, (int, float, np.number)) else str(r) for r in pivot.index]
    )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(value)

    plt.colorbar(im, ax=ax, label=value)
    return ax
