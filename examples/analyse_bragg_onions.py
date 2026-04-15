# %%
from __future__ import annotations

"""
examples/analyse_bragg_onions.py

Run a package-based parameter sweep for multilayer Bragg onion spheres and
generate a first set of screening figures.

What this script does
---------------------
1. Loads dispersive materials from local text files.
2. Runs a sweep over:
   - design peak wavelength
   - number of layers
   - outer layer
3. Computes scattering / backward-collected signal / colour metrics.
4. Saves raw and plot-friendly CSV tables.
5. Saves a few standard figures for screening candidate structures.

Requirements
------------
- Core package installed:
    python -m pip install -e .

- Optional simulation stack available:
    python -m pip install -e ".[scattnlay]"

You may also want SciPy installed to avoid colour-science warnings:
    conda install scipy
or:
    python -m pip install scipy

Usage
-----
From the repository root:

    python examples/analyse_bragg_onions.py

Important
---------
Before running this script, EDIT THE MATERIAL FILE PATHS below so they point
to your local refractive-index files.

Outputs
-------
Files are saved under:

    outputs/analyse_bragg_onions/

including:
- sweep_results_raw.csv
- sweep_results_plot.csv
- best_candidates.csv
- several PNG figures
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bragg_onion.materials import MaterialFileSpec, load_materials # type: ignore
from bragg_onion.sweep import ( # type: ignore
    run_bragg_onion_sweep,
    make_plotting_aliases,
    plot_sweep_metric,
    plot_sweep_heatmap,
    plot_sweep_colour_strip,
)


# ============================================================
# User settings
# ============================================================

# ------------------------------------------------------------
# EDIT THESE PATHS
# ------------------------------------------------------------
FILE_PS = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\PS(hPS)\x=0.0.txt"
FILE_P2VP = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\P2VP(PDP)\x=0.0.txt"
FILE_PMMA = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\PolymethylMethacrylate.txt"
FILE_H2O = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\Water_HaleQuerry.txt"

# ------------------------------------------------------------
# Sweep settings
# ------------------------------------------------------------
WAVELENGTHS_NM = np.arange(400, 701, 1)
THETA_DEG = np.linspace(0, 180, 361)

COLLECTION_NA = 0.8
COLLECTION_DIRECTION = "backward"
INTEGRATION_QUANTITY_FOR_COLOUR = "c_collected_m2"

PARAMETER_GRID = {
    "peak_wavelength_m": np.array([450e-9, 500e-9, 550e-9, 600e-9]),
    "n_layers": [7, 9, 11],
    "outer_layer": ["A", "B"],
    "core_thickness_factor": [0.5],
}

# Additional geometry kwargs applied to every case.
# Keep empty {} if not needed.
FIXED_GEOMETRY_KWARGS: dict = {}

# ------------------------------------------------------------
# Output settings
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "analyse_bragg_onions"


# ============================================================
# Helpers
# ============================================================

def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_current_figure(path: Path, dpi: int = 200) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def make_cielab_ab_scatter(df_plot: pd.DataFrame, save_path: Path) -> None:
    """
    Make a simple scatter plot in CIELAB a*–b* space coloured by the computed hex colour.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    for _, row in df_plot.iterrows():
        ax.scatter(
            row["a"],
            row["b"],
            color=row["hex"],
            s=120,
            edgecolor="black",
            linewidth=0.7,
        )

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("a*")
    ax.set_ylabel("b*")
    ax.set_title("Sweep results in CIELAB a*–b* plane")

    save_current_figure(save_path)


def print_best_candidates(df_plot: pd.DataFrame, n_top: int = 10) -> pd.DataFrame:
    """
    Print and return a ranked shortlist of candidate structures.
    """
    rank_cols = [
        "design_peak_wavelength_nm",
        "n_layers",
        "outer_layer",
        "diameter_nm",
        "eta_C",
        "eta_L",
        "eta_Y",
        "fraction_collected_max",
        "Ccollected_max_m2",
        "hex",
        "L",
        "a",
        "b",
        "C",
        "hue_deg",
    ]

    available_rank_cols = [c for c in rank_cols if c in df_plot.columns]

    best = df_plot.sort_values(
        by=["eta_C", "fraction_collected_max"],
        ascending=[False, False],
    )[available_rank_cols].head(n_top)

    print("\n=== Top candidate structures ===")
    print(best.to_string(index=False))

    return best


# ============================================================
# Main workflow
# ============================================================

def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    # --------------------------------------------------------
    # 1) Validate material file paths
    # --------------------------------------------------------
    for p in [FILE_PS, FILE_P2VP, FILE_H2O, FILE_PMMA]:
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Material file not found:\n{p}\n\n"
                "Please edit the FILE_PS / FILE_P2VP / FILE_H2O / FILE_PMMA paths in this script."
            )

    # --------------------------------------------------------
    # 2) Define grids
    # --------------------------------------------------------
    wavelengths_m = WAVELENGTHS_NM * 1e-9
    theta_rad = np.deg2rad(THETA_DEG)

    # --------------------------------------------------------
    # 3) Load materials
    # --------------------------------------------------------
    material_specs = {
        "PS": MaterialFileSpec(
            name="PS",
            path=FILE_PS,
            wavelength_unit="nm",
            skiprows=2,
            names=["Wavelength", "RefractiveIndex", "k"],
            n_column="RefractiveIndex",
            k_column="k",
            extrapolation="extrapolate",
        ),
        "P2VP": MaterialFileSpec(
            name="P2VP",
            path=FILE_P2VP,
            wavelength_unit="nm",
            skiprows=2,
            names=["Wavelength", "RefractiveIndex", "k"],
            n_column="RefractiveIndex",
            k_column="k",
            extrapolation="extrapolate",
        ),
        "H2O": MaterialFileSpec(
            name="H2O",
            path=FILE_H2O,
            wavelength_unit="um",
            skiprows=0,
            names=["Wavelength", "RefractiveIndex"],
            n_column="RefractiveIndex",
            k_column=None,
            extrapolation="extrapolate",
        ),
        "PMMA": MaterialFileSpec(
            name="PMMA",
            path=FILE_PMMA,
            wavelength_unit="nm",
            skiprows=2,
            names=["Wavelength", "RefractiveIndex", "k"],
            n_column="RefractiveIndex",
            k_column="k",
            extrapolation="extrapolate",
        ),
    }

    print("Loading materials...")
    loaded = load_materials(material_specs)

    mat_A = loaded["PS"]
    mat_B = loaded["P2VP"]
    medium = loaded["H2O"]
    mat_C = loaded["PMMA"]
    # --------------------------------------------------------
    # 4) Run the sweep
    # --------------------------------------------------------
    print("Running Bragg onion sweep...")
    sweep_result = run_bragg_onion_sweep(
        geometry_mode="peak_wavelength",
        parameter_grid=PARAMETER_GRID,
        material_a=mat_A,
        material_b=mat_B,
        n_medium=medium,
        wavelengths_m=wavelengths_m,
        theta_rad=theta_rad,
        collection_na=COLLECTION_NA,
        collection_direction=COLLECTION_DIRECTION,
        integration_quantity_for_colour=INTEGRATION_QUANTITY_FOR_COLOUR,
        colour_wavelength_min_nm=400.0,
        colour_wavelength_max_nm=700.0,
        colour_normalize_input=True,
        colour_normalization="max",
        fixed_geometry_kwargs=FIXED_GEOMETRY_KWARGS,
        store_full_results=False,
        progress=True,
    )

    df_raw = sweep_result.dataframe
    df_plot = make_plotting_aliases(df_raw)

    # --------------------------------------------------------
    # 5) Save tables
    # --------------------------------------------------------
    raw_csv = OUTPUT_DIR / "sweep_results_raw.csv"
    plot_csv = OUTPUT_DIR / "sweep_results_plot.csv"

    df_raw.to_csv(raw_csv, index=False)
    df_plot.to_csv(plot_csv, index=False)

    print(f"Saved raw sweep table to:   {raw_csv}")
    print(f"Saved plotting table to:    {plot_csv}")

    # --------------------------------------------------------
    # 6) Rank candidates
    # --------------------------------------------------------
    best = print_best_candidates(df_plot, n_top=10)
    best_csv = OUTPUT_DIR / "best_candidates.csv"
    best.to_csv(best_csv, index=False)
    print(f"Saved ranked shortlist to:  {best_csv}")

    # --------------------------------------------------------
    # 7) Figures
    # --------------------------------------------------------
    print("Generating figures...")

    # eta_C vs wavelength, split by outer layer
    for outer_layer in sorted(df_plot["outer_layer"].unique()):
        subset = df_plot[df_plot["outer_layer"] == outer_layer]

        plot_sweep_metric(
            subset,
            x="design_peak_wavelength_nm",
            y="eta_C",
            hue="n_layers",
            scale="linear",
        )
        plt.title(f"eta_C vs design wavelength (outer layer = {outer_layer})")
        save_current_figure(OUTPUT_DIR / f"eta_C_vs_design_peak_outer_{outer_layer}.png")

    # collected signal vs wavelength (dB), grouped by outer layer
    plot_sweep_metric(
        df_plot,
        x="design_peak_wavelength_nm",
        y="Ccollected_max_m2",
        hue="outer_layer",
        scale="db",
    )
    plt.title("Max backward collected signal vs design wavelength")
    save_current_figure(OUTPUT_DIR / "Ccollected_max_vs_design_peak_db.png")

    # heatmaps of eta_C for each outer layer
    for outer_layer in sorted(df_plot["outer_layer"].unique()):
        subset = df_plot[df_plot["outer_layer"] == outer_layer]

        plot_sweep_heatmap(
            subset,
            x="design_peak_wavelength_nm",
            y="n_layers",
            value="eta_C",
            scale="linear",
        )
        plt.title(f"eta_C heatmap (outer layer = {outer_layer})")
        save_current_figure(OUTPUT_DIR / f"eta_C_heatmap_outer_{outer_layer}.png")

    # colour strip
    plot_sweep_colour_strip(
        df_plot.sort_values(by=["design_peak_wavelength_nm", "n_layers", "outer_layer"]),
        x="design_peak_wavelength_nm",
        colour_hex_col="hex",
        label_col="n_layers",
    )
    plt.title("Computed colour strip (labels = n_layers)")
    save_current_figure(OUTPUT_DIR / "colour_strip.png")

    # CIELAB scatter
    make_cielab_ab_scatter(df_plot, OUTPUT_DIR / "cielab_ab_scatter.png")

    print(f"All outputs saved under:    {OUTPUT_DIR}")
    print("Analysis finished successfully.")


if __name__ == "__main__":
    main()