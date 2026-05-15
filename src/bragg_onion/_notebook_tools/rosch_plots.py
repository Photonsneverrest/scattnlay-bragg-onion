from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import bragg_onion

from bragg_onion.colour_solid_plotting import (
    load_rosch_macadam_colour_solid,
    plot_rosch_macadam_colour_solid,
    add_cielab_point,
)


FILLED_SYMBOLS = ["circle", "square", "diamond"]
LINE_SYMBOLS = ["cross", "x"]


def _load_optimal_colour_csv() -> pd.DataFrame:
    """
    Load the optimal-colour contour CSV from the installed bragg_onion package directory.
    """
    opt_csv_path = Path(bragg_onion.__file__).resolve().parent / "rosch_macadam_max_chroma_per_hue_1deg.csv"
    if not opt_csv_path.exists():
        raise FileNotFoundError(f"Optimal-colour CSV not found at: {opt_csv_path}")
    return pd.read_csv(opt_csv_path)


def _choose_colour_columns(df: pd.DataFrame, mode: str) -> dict[str, str]:
    if mode == "display":
        required_cols = ["L_display", "a_display", "b_display", "hex_display"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing display-normalized columns: {missing}. "
                "Run add_display_normalized_colour_columns(...) first."
            )
        return {
            "L_col": "L_display",
            "a_col": "a_display",
            "b_col": "b_display",
            "hex_col": "hex_display",
            "title_suffix": "display-normalized",
        }

    elif mode == "raw":
        required_cols = ["L", "a", "b", "hex"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing raw colour columns: {missing}")
        return {
            "L_col": "L",
            "a_col": "a",
            "b_col": "b",
            "hex_col": "hex",
            "title_suffix": "raw",
        }

    raise ValueError("mode must be 'raw' or 'display'.")


def _marker_line_kwargs(symbol: str, edge_color, edge_width):
    if symbol in FILLED_SYMBOLS:
        if edge_color is None:
            return dict(width=0)
        return dict(color=edge_color, width=edge_width)
    return dict(width=0)


def _build_symbol_map(series: pd.Series, marker_symbol_style: str) -> dict:
    if marker_symbol_style == "solid_only":
        valid_symbols = FILLED_SYMBOLS
    elif marker_symbol_style == "solid_and_line":
        valid_symbols = FILLED_SYMBOLS + LINE_SYMBOLS
    else:
        raise ValueError("marker_symbol_style must be 'solid_only' or 'solid_and_line'.")

    values = series.dropna().unique().tolist()
    try:
        values = sorted(values)
    except Exception:
        values = list(values)

    return {
        val: valid_symbols[i % len(valid_symbols)]
        for i, val in enumerate(values)
    }


def add_highlight_cielab_points(
    fig,
    highlight_df: pd.DataFrame,
    *,
    colour_view: str = "display",
    size: float = 9.0,
    symbol: str = "diamond",
    line_color: str = "white",
    line_width: float = 1.2,
):
    """
    Add a small number of highlighted CIELAB points to an existing Rosch–MacAdam figure.

    This uses add_cielab_point(...) and is intended only for small overlays
    (e.g. top 5 cases), not for the full sweep cloud.
    """
    cols = _choose_colour_columns(highlight_df, colour_view)
    L_col = cols["L_col"]
    a_col = cols["a_col"]
    b_col = cols["b_col"]
    hex_col = cols["hex_col"]

    out_fig = fig
    for _, row in highlight_df.iterrows():
        extra_hover = []
        for col in [
            "case_index",
            "design_peak_wavelength_nm",
            "n_layers",
            "outer_layer",
            "n_A",
            "k",
            "collection_angle_deg",
            "eta_C",
            "eta_L",
            "eta_Y",
        ]:
            if col in row.index:
                extra_hover.append(f"{col}: {row[col]}")

        out_fig = add_cielab_point(
            out_fig,
            L=float(row[L_col]),
            a=float(row[a_col]),
            b=float(row[b_col]),
            name=f"highlight {int(row['case_index'])}" if "case_index" in row.index else "highlight",
            color=str(row[hex_col]),
            size=size,
            symbol=symbol,
            line_color=line_color,
            line_width=line_width,
            extra_hover_lines=extra_hover,
        )
    return out_fig


def _build_rosch_macadam_figure(
    df_plot: pd.DataFrame,
    *,
    mode: str,
    max_overlay_points: int = 2000,
    fixed_marker_size: float = 4.0,
    marker_symbol_variable: str = "n_layers",
    marker_symbol_style: str = "solid_only",
    marker_edge_color="black",
    marker_edge_width: float = 0.4,
    optimal_marker_size: float = 4.8,
    optimal_marker_edge_color=None,
    optimal_marker_edge_width: float = 0.7,
):
    if marker_symbol_variable not in df_plot.columns:
        raise KeyError(
            f"marker_symbol_variable='{marker_symbol_variable}' is not a column in df_plot."
        )

    if len(df_plot) > max_overlay_points:
        df_overlay = (
            df_plot.sort_values(["eta_C", "eta_L", "eta_Y"], ascending=False)
                   .head(max_overlay_points)
                   .copy()
        )
    else:
        df_overlay = df_plot.copy()

    cols = _choose_colour_columns(df_overlay, mode)
    L_col = cols["L_col"]
    a_col = cols["a_col"]
    b_col = cols["b_col"]
    hex_col = cols["hex_col"]
    title_suffix = cols["title_suffix"]

    symbol_map = _build_symbol_map(df_overlay[marker_symbol_variable], marker_symbol_style)

    solid_df = load_rosch_macadam_colour_solid()
    fig = plot_rosch_macadam_colour_solid(
        solid_df,
        title=f"Bragg-onion cases in Rosch–MacAdam colour solid ({title_suffix})",
        max_points=50000,
        marker_size=1.2,
        opacity=0.05,
    )

    opt_df = _load_optimal_colour_csv()
    L_opt_col = "L_smooth" if "L_smooth" in opt_df.columns else "L"
    a_opt_col = "a_smooth" if "a_smooth" in opt_df.columns else "a"
    b_opt_col = "b_smooth" if "b_smooth" in opt_df.columns else "b"
    hex_opt_col = "hex_smooth" if "hex_smooth" in opt_df.columns else None

    # contour line
    fig.add_trace(
        go.Scatter3d(
            x=opt_df[a_opt_col],
            y=opt_df[b_opt_col],
            z=opt_df[L_opt_col],
            mode="lines",
            line=dict(color="black", width=5),
            name="optimal colour contour",
            hoverinfo="skip",
        )
    )

    # optimal-colour filled markers
    hover_opt = [
        f"hue={row['hue_deg']:.1f}°<br>"
        f"L={row[L_opt_col]:.2f}, a={row[a_opt_col]:.2f}, b={row[b_opt_col]:.2f}"
        for _, row in opt_df.iterrows()
    ]

    fig.add_trace(
        go.Scatter3d(
            x=opt_df[a_opt_col],
            y=opt_df[b_opt_col],
            z=opt_df[L_opt_col],
            mode="markers",
            marker=dict(
                size=optimal_marker_size,
                color=opt_df[hex_opt_col] if hex_opt_col is not None else "black",
                symbol="circle",
                line=_marker_line_kwargs("circle", optimal_marker_edge_color, optimal_marker_edge_width),
                opacity=1.0,
            ),
            name="optimal colours",
            hovertext=hover_opt,
            hoverinfo="text",
        )
    )

    # sweep cloud grouped by chosen marker variable
    for group_value, grp in df_overlay.groupby(marker_symbol_variable, sort=False):
        point_symbol = symbol_map.get(group_value, "circle")

        hovertext = []
        for _, row in grp.iterrows():
            lines = [
                f"case_index: {int(row['case_index'])}" if "case_index" in grp.columns else None,
                f"design λ: {row['design_peak_wavelength_nm']:.1f} nm" if "design_peak_wavelength_nm" in grp.columns else None,
                f"{marker_symbol_variable}: {row[marker_symbol_variable]}",
                f"n_layers: {int(row['n_layers'])}" if "n_layers" in grp.columns else None,
                f"outer_layer: {row['outer_layer']}" if "outer_layer" in grp.columns else None,
                f"n_A: {row['n_A']:.3f}" if "n_A" in grp.columns else None,
                f"k: {row['k']:.3f}" if "k" in grp.columns else None,
                f"collection angle: {row['collection_angle_deg']:.1f}°" if "collection_angle_deg" in grp.columns else None,
                f"eta_C: {row['eta_C']:.3f}" if "eta_C" in grp.columns else None,
                f"eta_L: {row['eta_L']:.3f}" if "eta_L" in grp.columns else None,
                f"eta_Y: {row['eta_Y']:.3f}" if "eta_Y" in grp.columns else None,
                f"q_collected_geom_max: {row['q_collected_geom_max']:.3f}" if "q_collected_geom_max" in grp.columns else None,
                f"hex: {row[hex_col]}",
            ]
            lines = [line for line in lines if line is not None]
            hovertext.append("<br>".join(lines))

        fig.add_trace(
            go.Scatter3d(
                x=grp[a_col],
                y=grp[b_col],
                z=grp[L_col],
                mode="markers",
                marker=dict(
                    size=fixed_marker_size,
                    color=grp[hex_col],
                    symbol=point_symbol,
                    line=_marker_line_kwargs(point_symbol, marker_edge_color, marker_edge_width),
                    opacity=0.95,
                ),
                name=f"{marker_symbol_variable} = {group_value}",
                hovertext=hovertext,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="a*", tickformat=".1f"),
            yaxis=dict(title="b*", tickformat=".1f"),
            zaxis=dict(title="L*", tickformat=".1f"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(itemsizing="constant"),
    )

    return fig, title_suffix


def make_rosch_macadam_figures(
    df_plot: pd.DataFrame,
    *,
    colour_view: str = "display",  # "raw", "display", "both"
    max_overlay_points: int = 2000,
    fixed_marker_size: float = 4.0,
    marker_symbol_variable: str = "n_layers",
    marker_symbol_style: str = "solid_only",
    marker_edge_color="black",
    marker_edge_width: float = 0.4,
    optimal_marker_size: float = 4.8,
    optimal_marker_edge_color=None,
    optimal_marker_edge_width: float = 0.7,
    save_html: bool = False,
    output_dir: Path | None = None,
    show_figures: bool = True,
) -> dict[str, go.Figure]:
    """
    Build Rosch–MacAdam figure(s).

    Returns a dict:
      - {"raw": fig}
      - {"display": fig}
      - or {"raw": fig_raw, "display": fig_display}
    """
    if colour_view == "both":
        modes_to_plot = ["raw", "display"]
    else:
        modes_to_plot = [colour_view]

    generated = {}

    for mode in modes_to_plot:
        fig, suffix = _build_rosch_macadam_figure(
            df_plot,
            mode=mode,
            max_overlay_points=max_overlay_points,
            fixed_marker_size=fixed_marker_size,
            marker_symbol_variable=marker_symbol_variable,
            marker_symbol_style=marker_symbol_style,
            marker_edge_color=marker_edge_color,
            marker_edge_width=marker_edge_width,
            optimal_marker_size=optimal_marker_size,
            optimal_marker_edge_color=optimal_marker_edge_color,
            optimal_marker_edge_width=optimal_marker_edge_width,
        )

        generated[mode] = fig

        if save_html:
            if output_dir is None:
                raise ValueError("output_dir must be provided if save_html=True")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            html_path = output_dir / f"rosch_macadam_colour_solid_{suffix}_with_optimal_colours.html"
            fig.write_html(str(html_path))
            print(f"Saved interactive 3D figure to: {html_path}")

        if show_figures:
            fig.show()

    return generated