from __future__ import annotations

# sweep.py
#
# Run parameter sweeps for Bragg onion scattering / colour simulations and
# visualize the results.
#
# Key features:
# - sweep arbitrary parameter grids via Cartesian product
# - supports geometry modes:
#   * "peak_wavelength"
#   * "thicknesses"
# - runs:
#   geometry -> stack -> solver -> integration -> colour conversion
# - collects results into a tidy pandas DataFrame
# - provides plotting helpers for quick assessment:
#   * line plots
#   * heatmaps
#   * colour strip plots
#
# Intended usage:
# - scan peak wavelength, number of layers, outer layer, NA, etc.
# - compare colour metrics and optical metrics across parameter space
#
# Notes:
# - Full result objects can optionally be stored, but this can consume memory.
# - The plotting functions are lightweight and operate on the DataFrame output.

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from .materials import Dispersion
from .geometry import (
    ExtinctionModifier,
    ExtraOuterShellSpec,
    build_bragg_onion_from_peak_wavelength,
    build_bragg_onion_from_thicknesses,
    resolve_layer_stack,
)

try:
    from .solver import ScatteringResult, run_scattnlay_spectrum
    from .integration import IntegratedScatteringResult, integrate_collection_na
    from .colour_adapter import ColourComputationResult, compute_colour_from_integrated_scattering
except ImportError as exc:
    raise ImportError(
        "bragg_onion.sweep requires the optional scattering stack "
        "(solver/integration/colour_adapter and ultimately scattnlay). "
        "Install it with: pip install 'bragg-onion[scattnlay]'"
    ) from exc


FloatArray = NDArray[np.float64]
GeometryMode = Literal["peak_wavelength", "thicknesses"]
ScaleMode = Literal["linear", "db"]


__all__ = [
    "SweepCaseResult",
    "SweepResult",
    "run_bragg_onion_sweep",
    "make_plotting_aliases",
    "plot_sweep_metric",
    "plot_sweep_heatmap",
    "plot_sweep_colour_strip",
]


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class SweepCaseResult:
    """
    Full result container for one sweep case.
    """
    case_index: int
    parameters: dict[str, Any]
    scattering_result: ScatteringResult
    integrated_result: IntegratedScatteringResult
    colour_result: ColourComputationResult


@dataclass
class SweepResult:
    """
    Container for a full sweep.

    Attributes
    ----------
    dataframe :
        Tidy pandas DataFrame with one row per parameter combination.
    case_results :
        Optional list of full per-case results (may be empty if not stored).
    geometry_mode :
        Geometry mode used for the sweep.
    integration_quantity :
        Quantity used for colour conversion.
    """
    dataframe: pd.DataFrame
    case_results: list[SweepCaseResult]
    geometry_mode: GeometryMode
    integration_quantity: str


# ============================================================
# Small helpers
# ============================================================

def _as_list(value: Any) -> list[Any]:
    """
    Ensure that a parameter-grid value is a list.
    Strings are treated as scalar values, not iterables.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return list(value.tolist())
    return [value]


def _cartesian_parameter_records(parameter_grid: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    """
    Expand a parameter grid into a list of parameter dictionaries.
    """
    keys = list(parameter_grid.keys())
    values = [_as_list(parameter_grid[k]) for k in keys]
    combos = product(*values)
    return [dict(zip(keys, combo)) for combo in combos]


def _flatten_nested_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """
    Flatten a nested dictionary using underscore-joined keys.

    Example
    -------
    {"CIELAB": {"L": 50, "a": 10}}
    ->
    {"CIELAB_L": 50, "CIELAB_a": 10}
    """
    flat: dict[str, Any] = {}
    for key, value in d.items():
        new_key = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_nested_dict(value, prefix=new_key))
        else:
            flat[new_key] = value
    return flat

def make_plotting_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the sweep DataFrame with shorter, plot-friendly aliases.

    This keeps the original columns intact in the input DataFrame and adds
    simpler names for the most commonly used plotting quantities.

    Added / renamed aliases include:
    - eta_C, eta_L, eta_Y
    - L, a, b, C, hue_deg
    - X, Y, Z
    - x, y, Y_xyY
    - r, g, b_rgb
    - h, s, v
    - hex
    """
    df_plot = df.copy()

    rename_map = {
        "Performance in Rosch-MacAdam Solid_eta_C": "eta_C",
        "Performance in Rosch-MacAdam Solid_eta_L": "eta_L",
        "Performance in Rosch-MacAdam Solid_eta_Y": "eta_Y",

        "CIELAB_L": "L",
        "CIELAB_a": "a",
        "CIELAB_b": "b",
        "CIELAB_C": "C",
        "CIELAB_hue_deg": "hue_deg",

        "XYZ_X": "X",
        "XYZ_Y": "Y",
        "XYZ_Z": "Z",

        "xyY_x": "x",
        "xyY_y": "y",
        "xyY_Y": "Y_xyY",

        "sRGB_r": "r",
        "sRGB_g": "g",
        "sRGB_b": "b_rgb",

        "HSV_h": "h",
        "HSV_s": "s",
        "HSV_v": "v",

        "Hex_hex": "hex",
    }

    existing_rename_map = {
        old: new for old, new in rename_map.items()
        if old in df_plot.columns
    }
    df_plot = df_plot.rename(columns=existing_rename_map)

    # Clean up wavelength columns
    for col in ["peak_wavelength_m", "design_peak_wavelength_m"]:
        if col in df_plot.columns:
            nm_col = col.replace("_m", "_nm")
            if nm_col not in df_plot.columns:
                df_plot[nm_col] = df_plot[col] * 1e9

    for nm_col in ["peak_wavelength_nm", "design_peak_wavelength_nm"]:
        if nm_col in df_plot.columns:
            vals = df_plot[nm_col].to_numpy(dtype=float)

            # If values are effectively integers, snap to integer-valued float
            if np.allclose(vals, np.round(vals), atol=1e-6):
                df_plot[nm_col] = np.round(vals).astype(float)
            else:
                df_plot[nm_col] = np.round(vals, 6)

    if "Warnings_messages" in df_plot.columns:
        df_plot["has_warnings"] = df_plot["Warnings_messages"].apply(
            lambda x: len(x) > 0 if isinstance(x, list) else bool(x)
        )

    # # Optional convenience columns:
    # if "peak_wavelength_m" in df_plot.columns and "peak_wavelength_nm" not in df_plot.columns:
    #     df_plot["peak_wavelength_nm"] = df_plot["peak_wavelength_m"] * 1e9

    # # Keep bool/string list columns more plot-friendly if desired
    # if "Warnings_messages" in df_plot.columns:
    #     df_plot["has_warnings"] = df_plot["Warnings_messages"].apply(
    #         lambda x: len(x) > 0 if isinstance(x, list) else bool(x)
    #     )

    return df_plot


def _safe_trapz(y: np.ndarray, x: np.ndarray) -> float:
    """
    Trapezoidal integral with basic safety checks.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 2 or x.size < 2:
        return 0.0
    return float(np.trapz(y, x))


def _spectral_summary(
    wavelengths_nm: FloatArray,
    spectrum: FloatArray,
) -> dict[str, float]:
    """
    Compute compact summary metrics of a spectrum.
    """
    wl = np.asarray(wavelengths_nm, dtype=float)
    spec = np.asarray(spectrum, dtype=float)

    if wl.ndim != 1 or spec.ndim != 1 or wl.shape != spec.shape:
        raise ValueError("wavelengths_nm and spectrum must be 1D arrays of identical shape.")

    if spec.size == 0:
        return {
            "spectrum_peak_wavelength_nm": np.nan,
            "spectrum_peak_value": np.nan,
            "spectrum_integral": np.nan,
            "spectrum_centroid_nm": np.nan,
        }

    idx_peak = int(np.argmax(spec))
    peak_wavelength_nm = float(wl[idx_peak])
    peak_value = float(spec[idx_peak])
    integral = _safe_trapz(spec, wl)

    if np.sum(spec) > 0:
        centroid_nm = float(np.sum(wl * spec) / np.sum(spec))
    else:
        centroid_nm = np.nan

    return {
        "spectrum_peak_wavelength_nm": peak_wavelength_nm,
        "spectrum_peak_value": peak_value,
        "spectrum_integral": integral,
        "spectrum_centroid_nm": centroid_nm,
    }


def _apply_scale(values: np.ndarray, scale: ScaleMode, floor: float = 1e-30) -> np.ndarray:
    """
    Apply linear or dB scaling.
    """
    arr = np.asarray(values, dtype=float)
    if scale == "linear":
        return arr
    if scale == "db":
        return 10.0 * np.log10(np.maximum(arr, floor))
    raise ValueError(f"Unknown scale: {scale!r}")


# ============================================================
# Main sweep execution
# ============================================================

def run_bragg_onion_sweep(
    *,
    geometry_mode: GeometryMode,
    parameter_grid: dict[str, Iterable[Any]],
    material_a: Dispersion,
    material_b: Dispersion,
    n_medium: float | complex | Dispersion,
    wavelengths_m: FloatArray,
    theta_rad: FloatArray,
    collection_na: float,
    collection_direction: Literal["forward", "backward"] = "backward",
    integration_quantity_for_colour: Literal[
        "c_collected_m2",
        "fraction_collected",
        "q_collected_geom",
    ] = "q_collected_geom",
    colour_wavelength_min_nm: float = None,
    colour_wavelength_max_nm: float = None,
    colour_normalize_input: bool = False,
    colour_normalization: Literal["max", "sum"] = "max",
    fixed_geometry_kwargs: dict[str, Any] | None = None,
    extinction_modifiers: ExtinctionModifier | list[ExtinctionModifier] | None = None,
    extra_outer_shell: ExtraOuterShellSpec | None = None,
    store_full_results: bool = False,
    progress: bool = True,
) -> SweepResult:
    """
    Run a Cartesian parameter sweep for Bragg onion simulations.

    Parameters
    ----------
    geometry_mode :
        "peak_wavelength" or "thicknesses"
    parameter_grid :
        Dictionary of parameters to sweep.
        Example:
            {"peak_wavelength_m": [...], "n_layers": [...], "outer_layer": ["A", "B"]}
        or
            {"t_a_m": [...], "t_b_m": [...], "n_layers": [...], "outer_layer": ["A", "B"]}
    material_a, material_b :
        Base Bragg materials
    n_medium :
        Surrounding medium refractive index
    wavelengths_m :
        Wavelength grid for the scattering simulation
    theta_rad :
        Angular grid for the scattering simulation
    collection_na :
        Numerical aperture for collection integration
    collection_direction :
        "forward" or "backward"
    integration_quantity_for_colour :
        Which integrated spectrum should feed colour conversion
    colour_wavelength_min_nm, colour_wavelength_max_nm :
        Wavelength range used for colour conversion
    colour_normalize_input :
        Passed to compute_color_properties() via colour_adapter
    colour_normalization :
        Passed to compute_color_properties() via colour_adapter
    fixed_geometry_kwargs :
        Additional geometry arguments held fixed for all sweep cases
    extinction_modifiers :
        Optional extinction modifiers applied to every case
    extra_outer_shell :
        Optional extra outer shell applied to every case
    store_full_results :
        If True, keep full per-case result objects
    progress :
        If True, print progress information

    Returns
    -------
    SweepResult
    """
    fixed_geometry_kwargs = dict(fixed_geometry_kwargs or {})
    parameter_records = _cartesian_parameter_records(parameter_grid)

    rows: list[dict[str, Any]] = []
    case_results: list[SweepCaseResult] = []

    for case_index, params in enumerate(parameter_records):
        if progress:
            print(f"[sweep] case {case_index + 1}/{len(parameter_records)}: {params}")

        # --------------------------------------------------------
        # 1) Build geometry
        # --------------------------------------------------------
        geom_kwargs = dict(fixed_geometry_kwargs)
        geom_kwargs.update(params)

        if geometry_mode == "peak_wavelength":
            geometry = build_bragg_onion_from_peak_wavelength(
                material_a=material_a,
                material_b=material_b,
                extra_outer_shell=extra_outer_shell,
                **geom_kwargs,
            )
        elif geometry_mode == "thicknesses":
            geometry = build_bragg_onion_from_thicknesses(
                extra_outer_shell=extra_outer_shell,
                **geom_kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported geometry_mode: {geometry_mode!r}. "
                "Use 'peak_wavelength' or 'thicknesses'."
            )

        # --------------------------------------------------------
        # 2) Resolve layer stack
        # --------------------------------------------------------
        stack = resolve_layer_stack(
            geometry=geometry,
            material_a=material_a,
            material_b=material_b,
            extinction_modifiers=extinction_modifiers,
        )

        # --------------------------------------------------------
        # 3) Run scattering solver
        # --------------------------------------------------------
        scattering_result = run_scattnlay_spectrum(
            stack=stack,
            wavelengths_m=wavelengths_m,
            theta_rad=theta_rad,
            n_medium=n_medium,
        )

        # --------------------------------------------------------
        # 4) Integrate over collection NA
        # --------------------------------------------------------
        integrated_result = integrate_collection_na(
            scattering_result,
            collection_na=collection_na,
            direction=collection_direction,
        )

        # --------------------------------------------------------
        # 5) Compute colour
        # --------------------------------------------------------
        colour_result = compute_colour_from_integrated_scattering(
            integrated_result,
            quantity=integration_quantity_for_colour,
            wavelength_min_nm=colour_wavelength_min_nm,
            wavelength_max_nm=colour_wavelength_max_nm,
            normalize_input=colour_normalize_input,
            normalization=colour_normalization,
        )

        # --------------------------------------------------------
        # 6) Summarize into one DataFrame row
        # --------------------------------------------------------

        
        q_geom = integrated_result.q_collected_geom
        wl_nm = integrated_result.wavelengths_m * 1e9

        row: dict[str, Any] = {
            "case_index": case_index,
            "geometry_mode": geometry_mode,
            "collection_na": float(collection_na),
            "collection_direction": collection_direction,
            "integration_quantity_for_colour": integration_quantity_for_colour,
            "n_layers": int(geometry.n_layers),
            "n_layers_total": int(geometry.n_layers_total),
            "outer_layer": geometry.outer_layer,
            "actual_outer_layer": geometry.actual_outer_layer,
            "diameter_nm": float(geometry.diameter_m * 1e9),
            "outer_radius_nm": float(geometry.outer_radius_m * 1e9),
            "core_radius_nm": float(geometry.core_radius_m * 1e9),
            "t_a_nm": float(geometry.t_a_m * 1e9),
            "t_b_nm": float(geometry.t_b_m * 1e9),
            "core_thickness_factor": float(geometry.core_thickness_factor),
            "q_collected_geom_max": float(np.max(q_geom)),
            "q_collected_geom_peak_wavelength_nm": float(
                wl_nm[np.argmax(q_geom)]
            ),

        }

        if geometry.design_peak_wavelength_m is not None:
            row["design_peak_wavelength_nm"] = float(geometry.design_peak_wavelength_m * 1e9)

        # keep explicit sweep params in the row too
        row.update(_flatten_nested_dict(colour_result.color_properties))
        row.update(params)

        # basic scattering summaries
        row["Qsca_max"] = float(np.max(scattering_result.qsca))
        row["Qsca_peak_wavelength_nm"] = float(
            scattering_result.wavelengths_m[np.argmax(scattering_result.qsca)] * 1e9
        )
        row["Qext_max"] = float(np.max(scattering_result.qext))
        row["Qabs_max"] = float(np.max(scattering_result.qabs))
        row["Csca_max_m2"] = float(np.max(scattering_result.csca_m2))
        row["Ccollected_max_m2"] = float(np.max(integrated_result.c_collected_m2))
        row["fraction_collected_max"] = float(np.max(integrated_result.fraction_collected))
        row["fraction_collected_mean"] = float(np.mean(integrated_result.fraction_collected))

        # spectral summary of the colour-driving spectrum
        row.update(
            _spectral_summary(
                colour_result.wavelengths_nm,
                colour_result.spectrum_used,
            )
        )

        # flatten colour properties into columns
        flat_colour = _flatten_nested_dict(colour_result.color_properties)
        row.update(flat_colour)

        rows.append(row)

        if store_full_results:
            case_results.append(
                SweepCaseResult(
                    case_index=case_index,
                    parameters=dict(params),
                    scattering_result=scattering_result,
                    integrated_result=integrated_result,
                    colour_result=colour_result,
                )
            )

    df = pd.DataFrame(rows)
    return SweepResult(
        dataframe=df,
        case_results=case_results,
        geometry_mode=geometry_mode,
        integration_quantity=integration_quantity_for_colour,
    )


# ============================================================
# Plotting helpers for sweep assessment
# ============================================================

def plot_sweep_metric(
    sweep_df: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str | None = None,
    scale: ScaleMode = "linear",
    floor: float = 1e-30,
    sort_x: bool = True,
    marker: str = "o",
    ax: plt.Axes | None = None,
):
    """
    Plot a sweep metric versus one parameter, optionally grouped by hue.

    Example
    -------
    plot_sweep_metric(df, x="design_peak_wavelength_nm", y="Performance in Rosch-MacAdam Solid_eta_C", hue="n_layers")
    """
    if x not in sweep_df.columns:
        raise ValueError(f"Column '{x}' not found in DataFrame.")
    if y not in sweep_df.columns:
        raise ValueError(f"Column '{y}' not found in DataFrame.")
    if hue is not None and hue not in sweep_df.columns:
        raise ValueError(f"Column '{hue}' not found in DataFrame.")

    if ax is None:
        fig, ax = plt.subplots()

    if hue is None:
        df_plot = sweep_df.copy()
        if sort_x:
            df_plot = df_plot.sort_values(by=x)
        y_plot = _apply_scale(df_plot[y].to_numpy(dtype=float), scale=scale, floor=floor)
        ax.plot(df_plot[x], y_plot, marker=marker)
    else:
        for group_value, df_group in sweep_df.groupby(hue):
            df_plot = df_group.copy()
            if sort_x:
                df_plot = df_plot.sort_values(by=x)
            y_plot = _apply_scale(df_plot[y].to_numpy(dtype=float), scale=scale, floor=floor)
            ax.plot(df_plot[x], y_plot, marker=marker, label=str(group_value))
        ax.legend(title=hue)

    ax.set_xlabel(x)
    ax.set_ylabel(f"{y} ({scale})" if scale == "db" else y)
    ax.set_title(f"{y} vs {x}")
    return ax


def plot_sweep_heatmap(
    sweep_df: pd.DataFrame,
    *,
    x: str,
    y: str,
    value: str,
    scale: ScaleMode = "linear",
    floor: float = 1e-30,
    aggfunc: str = "mean",
    cmap: str = "viridis",
    ax: plt.Axes | None = None,
):
    """
    Plot a 2D heatmap of one metric over two sweep parameters.

    Example
    -------
    plot_sweep_heatmap(df, x="design_peak_wavelength_nm", y="n_layers", value="Performance in Rosch-MacAdam Solid_eta_C")
    """
    for col in (x, y, value):
        if col not in sweep_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    pivot = pd.pivot_table(
        sweep_df,
        index=y,
        columns=x,
        values=value,
        aggfunc=aggfunc,
    )

    values = pivot.to_numpy(dtype=float)
    values_plot = _apply_scale(values, scale=scale, floor=floor)

    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(
        values_plot,
        aspect="auto",
        origin="lower",
        cmap=cmap,
    )
    
    x_labels = []
    for v in pivot.columns:
        if isinstance(v, (int, np.integer)):
            x_labels.append(f"{v:d}")
        elif isinstance(v, (float, np.floating)):
            if np.isclose(v, round(v), atol=1e-6):
                x_labels.append(f"{int(round(v))}")
            else:
                x_labels.append(f"{v:.1f}")
        else:
            x_labels.append(str(v))

    y_labels = []
    for v in pivot.index:
        if isinstance(v, (int, np.integer)):
            y_labels.append(f"{v:d}")
        elif isinstance(v, (float, np.floating)):
            if np.isclose(v, round(v), atol=1e-6):
                y_labels.append(f"{int(round(v))}")
            else:
                y_labels.append(f"{v:.1f}")
        else:
            y_labels.append(str(v))

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(y_labels)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{value} heatmap")

    cbar_label = f"{value} ({scale})" if scale == "db" else value
    plt.colorbar(im, ax=ax, label=cbar_label)

    return ax



def plot_sweep_colour_strip(
    sweep_df: pd.DataFrame,
    *,
    x: str,
    colour_hex_col: str = "Hex_hex",
    sort_by: str | None = None,
    label_col: str | None = None,
    ax: plt.Axes | None = None,
):
    """
    Plot a 1D colour strip using computed hex colours.

    Example
    -------
    plot_sweep_colour_strip(df, x="design_peak_wavelength_nm")
    """
    if x not in sweep_df.columns:
        raise ValueError(f"Column '{x}' not found in DataFrame.")
    if colour_hex_col not in sweep_df.columns:
        raise ValueError(f"Column '{colour_hex_col}' not found in DataFrame.")

    df_plot = sweep_df.copy()
    if sort_by is not None:
        if sort_by not in df_plot.columns:
            raise ValueError(f"Column '{sort_by}' not found in DataFrame.")
        df_plot = df_plot.sort_values(by=sort_by)
    else:
        df_plot = df_plot.sort_values(by=x)

    x_vals = df_plot[x].to_list()
    hex_vals = df_plot[colour_hex_col].astype(str).to_list()

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(x_vals) * 0.45), 1.8))

    ax.set_xlim(0, len(x_vals))
    ax.set_ylim(0, 1)

    
    for i, hex_val in enumerate(hex_vals):
        rect = Rectangle((i, 0), 1, 1, facecolor=hex_val, edgecolor="black", linewidth=0.5)
        ax.add_patch(rect)

        if label_col is not None:
            if label_col not in df_plot.columns:
                raise ValueError(f"Column '{label_col}' not found in DataFrame.")
            label = str(df_plot.iloc[i][label_col])
            ax.text(i + 0.5, 0.5, label, ha="center", va="center", fontsize=8)

    # Clean x tick labels
    x_labels = []
    for v in x_vals:
        if isinstance(v, (int, np.integer)):
            x_labels.append(f"{v:d}")
        elif isinstance(v, (float, np.floating)):
            if np.isclose(v, round(v), atol=1e-6):
                x_labels.append(f"{int(round(v))}")
            else:
                x_labels.append(f"{v:.1f}")
        else:
            x_labels.append(str(v))

    ax.set_xticks(np.arange(len(x_vals)) + 0.5)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_yticks([])
    ax.set_xlabel(x)
    ax.set_title("Computed colour strip")

    return ax