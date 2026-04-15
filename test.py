# %%
# Bragg Onion Scattnlay testing

import sys
sys.path.append('C:/Users/SchwarzN/OneDrive - Université de Fribourg/Institution/P1_BraggSpericalPigments/Simulation/MIE_Computation/Scattnlay_BraggOnion/src/bragg_onion')

# %%
# materials testing
import materials # type: ignore

import importlib
importlib.reload(materials)

import numpy as np

from materials import MaterialFileSpec, load_materials

# Define visible wavelength range in nm
visible_wavelengths_nm = np.arange(350, 700, 1)

# File paths
file_TiO2 = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\TiO2_Bodurov.txt"
file_P2VP = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\P2VP(PDP)\x=0.0.txt"
file_PS   = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\PS(hPS)\x=0.0.txt"
file_H2O  = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\Water_HaleQuerry.txt"

# Define how each material file should be parsed
material_specs = {
    "TiO2": MaterialFileSpec(
        name="TiO2",
        path=file_TiO2,
        wavelength_unit="um",
        skiprows=0,
        names=["Wavelength", "RefractiveIndex"],
        n_column="RefractiveIndex",
        k_column=None,
        extrapolation="extrapolate",
    ),
    "P2VP": MaterialFileSpec(
        name="P2VP",
        path=file_P2VP,
        wavelength_unit="nm",
        skiprows=2,
        names=["Wavelength", "RefractiveIndex", "k"],
        n_column="RefractiveIndex",
        k_column="k",
        extrapolation="extrapolate",
    ),
    "PS": MaterialFileSpec(
        name="PS",
        path=file_PS,
        wavelength_unit="nm",
        skiprows=2,
        names=["Wavelength", "RefractiveIndex", "k"],
        n_column="RefractiveIndex",
        k_column="k",
        extrapolation="extrapolate",
    ),
    "H2O": MaterialFileSpec(
        name="H2O",
        path=file_H2O,
        wavelength_unit="um",
        skiprows=0,
        names=["Wavelength", "RefractiveIndex"],
        n_column="RefractiveIndex",
        k_column=None,
        extrapolation="extrapolate",
    ),
}

# Load materials
materials = load_materials(material_specs)

# Convert visible wavelengths to meters for internal use
visible_wavelengths_m = visible_wavelengths_nm * 1e-9

# Evaluate refractive indices on your target wavelength grid
refractive_indices = {
    "TiO2": materials["TiO2"](visible_wavelengths_m),
    "P2VP": materials["P2VP"](visible_wavelengths_m),
    "PS": materials["PS"](visible_wavelengths_m),
    "H2O": materials["H2O"](visible_wavelengths_m),
}
# %%
# geometry testing
import numpy as np
import importlib

import materials
import geometry

importlib.reload(materials)
importlib.reload(geometry)

from materials import MaterialFileSpec, load_materials
from geometry import (
    ExtinctionModifier,
    ExtraOuterShellSpec,
    build_bragg_onion_from_peak_wavelength,
    build_bragg_onion_from_thicknesses,
    resolve_layer_stack,
)

# ------------------------------------------------------------
# 1) Define wavelength grid
# ------------------------------------------------------------
wavelengths_nm = np.arange(400, 701, 1)
wavelengths_m = wavelengths_nm * 1e-9

# ------------------------------------------------------------
# 2) Load materials
# ------------------------------------------------------------
file_TiO2 = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\TiO2_Bodurov.txt"
file_P2VP = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\P2VP(PDP)\x=0.0.txt"
file_PS   = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\PS(hPS)\x=0.0.txt"
file_H2O  = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\Water_HaleQuerry.txt"

material_specs = {
    "TiO2": MaterialFileSpec(
        name="TiO2",
        path=file_TiO2,
        wavelength_unit="um",
        skiprows=0,
        names=["Wavelength", "RefractiveIndex"],
        n_column="RefractiveIndex",
        k_column=None,
        extrapolation="extrapolate",
    ),
    "PS": MaterialFileSpec(
        name="PS",
        path=file_PS,
        wavelength_unit="nm",
        skiprows=2,
        names=["Wavelength", "RefractiveIndex", "k"],
        n_column="RefractiveIndex",
        k_column="k",
        extrapolation="extrapolate",
    ),
    "P2VP": MaterialFileSpec(
        name="P2VP",
        path=file_P2VP,
        wavelength_unit="nm",
        skiprows=2,
        names=["Wavelength", "RefractiveIndex", "k"],
        n_column="RefractiveIndex",
        k_column="k",
        extrapolation="extrapolate",
    ),
    "H2O": MaterialFileSpec(
        name="H2O",
        path=file_H2O,
        wavelength_unit="um",
        skiprows=0,
        names=["Wavelength", "RefractiveIndex"],
        n_column="RefractiveIndex",
        k_column=None,
        extrapolation="extrapolate",
    ),
}

loaded = load_materials(material_specs)

# Choose two Bragg materials
mat_A = loaded["PS"]
mat_B = loaded["P2VP"]
medium = loaded["H2O"]

# ------------------------------------------------------------
# 3) Optional extra outer shell
# ------------------------------------------------------------
extra_shell = ExtraOuterShellSpec(
    thickness_m=30e-9,
    material=loaded["TiO2"],
    name="TiO2_outer_shell",
)

# ------------------------------------------------------------
# 4) Optional extinction modifications
# ------------------------------------------------------------
core_absorber = ExtinctionModifier(
    target="core",
    mode="add",
    profile=0.01,
    name="core_constant_absorption",
)

all_A_substitute = ExtinctionModifier(
    target="material_A",
    mode="substitute",
    profile=lambda wl_m: 0.002 + 0.01 * np.exp(-((wl_m * 1e9 - 550.0) / 50.0) ** 2),
    name="A_gaussian_absorption",
)

all_B_add = ExtinctionModifier(
    target="material_B",
    mode="add",
    profile= 0.007,
    name="B_constant_absorption",
)

modifiers = [core_absorber, all_A_substitute, all_B_add]

# ------------------------------------------------------------
# 5) Build quarter-wave Bragg onion
# ------------------------------------------------------------
peak_wavelength_m = 550e-9

geom = build_bragg_onion_from_peak_wavelength(
    material_a=mat_A,
    material_b=mat_B,
    peak_wavelength_m=peak_wavelength_m,
    outer_layer="A",
    n_layers=9,
    extra_outer_shell=extra_shell,
    core_thickness_factor=1,   # matches your earlier example if desired
)

print("=== Geometry ===")
print("layer labels:", geom.layer_labels)
print("layer thicknesses [nm]:", geom.layer_thicknesses_m * 1e9)
print("radii [nm]:", geom.radii_m * 1e9)
print("diameter [nm]:", geom.diameter_m * 1e9)
print("n_layers (Bragg only):", geom.n_layers)
print("n_layers_total:", geom.n_layers_total)

# ------------------------------------------------------------
# 5b) Build Bragg onion with custom thicknesses
# ------------------------------------------------------------

t_a_m = 60e-9
t_b_m = 40e-9

geom_custom = build_bragg_onion_from_thicknesses(
    t_a_m=t_a_m,
    t_b_m=t_b_m,
    outer_layer="B",
    diameter_m=500e-9,
)
print("\n=== Custom Geometry ===")
print("layer labels:", geom_custom.layer_labels)
print("layer thicknesses [nm]:", geom_custom.layer_thicknesses_m * 1e9)
print("radii [nm]:", geom_custom.radii_m * 1e9)
print("diameter [nm]:", geom_custom.diameter_m * 1e9)
print("n_layers (Bragg only):", geom_custom.n_layers)
print("n_layers_total:", geom_custom.n_layers_total)


# ------------------------------------------------------------
# 6) Resolve effective layer stack
# ------------------------------------------------------------
stack = resolve_layer_stack(
    geometry=geom,
    material_a=mat_A,
    material_b=mat_B,
    extinction_modifiers=modifiers,
)

print("\n=== Resolved materials ===")
print("material names:", stack.layer_material_names)

# ------------------------------------------------------------
# 7) Compute refractive indices and m
# ------------------------------------------------------------
nk_550 = stack.refractive_indices_at_wavelength(550e-9)
m_550 = stack.m_at_wavelength(550e-9, n_medium=medium)
m_spectrum = stack.m_spectrum(wavelengths_m, n_medium=medium)

print("\n=== Optical arrays ===")
print("nk at 550 nm:", nk_550)
print("m at 550 nm:", m_550)
print("shape of m_spectrum:", m_spectrum.shape)   # should be (len(wavelengths_m), n_layers_total)
# %%
# test solver
import numpy as np
import importlib

import materials
import geometry
import solver

importlib.reload(materials)
importlib.reload(geometry)
importlib.reload(solver)

from materials import MaterialFileSpec, load_materials
from geometry import build_bragg_onion_from_peak_wavelength, resolve_layer_stack
from solver import run_scattnlay_spectrum

# ------------------------------------------------------------
# 1) Wavelength and angular grids
# ------------------------------------------------------------
wavelengths_nm = np.arange(450, 701, 10)
wavelengths_m = wavelengths_nm * 1e-9

theta_deg = np.linspace(0, 180, 181)
theta_rad = np.deg2rad(theta_deg)

# ------------------------------------------------------------
# 2) Load materials
# ------------------------------------------------------------
file_PS   = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\PS(hPS)\x=0.0.txt"
file_P2VP = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\P2VP(PDP)\x=0.0.txt"
file_H2O  = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\Water_HaleQuerry.txt"

material_specs = {
    "PS": MaterialFileSpec(
        name="PS",
        path=file_PS,
        wavelength_unit="nm",
        skiprows=2,
        names=["Wavelength", "RefractiveIndex", "k"],
        n_column="RefractiveIndex",
        k_column="k",
        extrapolation="extrapolate",
    ),
    "P2VP": MaterialFileSpec(
        name="P2VP",
        path=file_P2VP,
        wavelength_unit="nm",
        skiprows=2,
        names=["Wavelength", "RefractiveIndex", "k"],
        n_column="RefractiveIndex",
        k_column="k",
        extrapolation="extrapolate",
    ),
    "H2O": MaterialFileSpec(
        name="H2O",
        path=file_H2O,
        wavelength_unit="um",
        skiprows=0,
        names=["Wavelength", "RefractiveIndex"],
        n_column="RefractiveIndex",
        k_column=None,
        extrapolation="extrapolate",
    ),
}

loaded = load_materials(material_specs)

mat_A = loaded["PS"]
mat_B = loaded["P2VP"]
medium = loaded["H2O"]

# ------------------------------------------------------------
# 3) Build geometry
# ------------------------------------------------------------
geom = build_bragg_onion_from_peak_wavelength(
    material_a=mat_A,
    material_b=mat_B,
    peak_wavelength_m=550e-9,
    outer_layer="A",
    n_layers=9,
    core_thickness_factor=0.5,   # if you want to match your earlier convention
)

stack = resolve_layer_stack(
    geometry=geom,
    material_a=mat_A,
    material_b=mat_B,
)

# ------------------------------------------------------------
# 4) Run scattnlay
# ------------------------------------------------------------
result = run_scattnlay_spectrum(
    stack=stack,
    wavelengths_m=wavelengths_m,
    theta_rad=theta_rad,
    n_medium=medium,
)

# ------------------------------------------------------------
# 5) Inspect results
# ------------------------------------------------------------
print("radii [nm]:", result.radii_m * 1e9)
print("x shape:", result.x.shape)
print("m shape:", result.m.shape)
print("S1 shape:", result.s1.shape)
print("dcs_m2_sr shape:", result.dcs_m2_sr.shape)
print("qsca shape:", result.qsca.shape)

# Example: differential scattering at 550 nm
idx_550 = np.argmin(np.abs(wavelengths_nm - 550))
print("Qsca at 550 nm:", result.qsca[idx_550])
print("Csca at 550 nm [m^2]:", result.csca_m2[idx_550])
print("Backscattering angle value at 180 deg [m^2/sr]:", result.dcs_m2_sr[idx_550, -1])
# %%
import numpy as np
import importlib

import integration
importlib.reload(integration)

from integration import integrate_collection_na, integrate_theta_range

# Example 1: forward collection with NA = 0.8
forward_na = integrate_collection_na(
    result,
    collection_na=0.8,
    direction="forward",
)

print("Forward collected Csca [m^2] at 550 nm:",
      forward_na.c_collected_m2[np.argmin(np.abs(wavelengths_nm - 550))])

print("Forward collected fraction at 550 nm:",
      forward_na.fraction_collected[np.argmin(np.abs(wavelengths_nm - 550))])

# Example 2: backward hemisphere
backward_hemi = integrate_theta_range(
    result,
    theta_min_rad=np.pi / 2,
    theta_max_rad=np.pi,
    direction="backward_hemisphere",
)

print("Backward hemisphere fraction at 550 nm:",
      backward_hemi.fraction_collected[np.argmin(np.abs(wavelengths_nm - 550))])
# %%
import importlib
import plotting
importlib.reload(plotting)

import matplotlib.pyplot as plt
from plotting import (
    plot_differential_scattering_vs_wavelength,
    plot_integrated_scattering,
    plot_scattering_polar,
    plot_efficiency_vs_wavelength,
)

# Differential scattering at selected angles
plot_differential_scattering_vs_wavelength(
    result,
    angles_deg=[0, 30, 60, 90, 120, 150, 180],
    quantity="dcs_m2_sr",
    scale="linear",
)
plt.show()

# Same in dB
plot_differential_scattering_vs_wavelength(
    result,
    angles_deg=[0, 30, 60, 90, 120, 150, 180],
    quantity="dcs_m2_sr",
    scale="db",
)
plt.show()

# Scalar efficiency
plot_efficiency_vs_wavelength(
    result,
    quantity="qsca",
    scale="linear",
)
plt.show()

# Backward NA-collected scattering (reflectance-like proxy)
backward_na = integrate_collection_na(result, collection_na=0.8, direction="backward")

plot_integrated_scattering(
    backward_na,
    quantity="fraction_collected",
    scale="linear",
)
plt.show()

# Polar plot at 550 nm
plot_scattering_polar(
    result,
    wavelength_m=550e-9,
    quantity="dcs_geom_norm_sr_inv",
    scale="db",
)
plt.show()
# %%
import numpy as np
import importlib

import materials
import geometry
import solver
import integration
import spectrum_colour_props

importlib.reload(materials)
importlib.reload(geometry)
importlib.reload(solver)
importlib.reload(integration)
importlib.reload(spectrum_colour_props)

from materials import MaterialFileSpec, load_materials
from geometry import build_bragg_onion_from_peak_wavelength, resolve_layer_stack
from solver import run_scattnlay_spectrum
from integration import integrate_collection_na
from spectrum_colour_props import compute_color_properties

# ------------------------------------------------------------
# 1) Define wavelength and angle grids
# ------------------------------------------------------------
wavelengths_nm = np.arange(400, 701, 1)
wavelengths_m = wavelengths_nm * 1e-9

theta_deg = np.linspace(0, 180, 361)
theta_rad = np.deg2rad(theta_deg)

# ------------------------------------------------------------
# 2) Load materials
# ------------------------------------------------------------
file_PS   = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\PS(hPS)\x=0.0.txt"
file_P2VP = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\P2VP(PDP)\x=0.0.txt"
file_H2O  = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\Water_HaleQuerry.txt"

material_specs = {
    "PS": MaterialFileSpec(
        name="PS",
        path=file_PS,
        wavelength_unit="nm",
        skiprows=2,
        names=["Wavelength", "RefractiveIndex", "k"],
        n_column="RefractiveIndex",
        k_column="k",
        extrapolation="extrapolate",
    ),
    "P2VP": MaterialFileSpec(
        name="P2VP",
        path=file_P2VP,
        wavelength_unit="nm",
        skiprows=2,
        names=["Wavelength", "RefractiveIndex", "k"],
        n_column="RefractiveIndex",
        k_column="k",
        extrapolation="extrapolate",
    ),
    "H2O": MaterialFileSpec(
        name="H2O",
        path=file_H2O,
        wavelength_unit="um",
        skiprows=0,
        names=["Wavelength", "RefractiveIndex"],
        n_column="RefractiveIndex",
        k_column=None,
        extrapolation="extrapolate",
    ),
}

loaded = load_materials(material_specs)

mat_A = loaded["PS"]
mat_B = loaded["P2VP"]
medium = loaded["H2O"]

# ------------------------------------------------------------
# 3) Build geometry and resolve layer stack
# ------------------------------------------------------------
geom = build_bragg_onion_from_peak_wavelength(
    material_a=mat_A,
    material_b=mat_B,
    peak_wavelength_m=550e-9,
    outer_layer="A",
    n_layers=9,
    core_thickness_factor=0.5,
)

stack = resolve_layer_stack(
    geometry=geom,
    material_a=mat_A,
    material_b=mat_B,
)

# ------------------------------------------------------------
# 4) Run scattering simulation
# ------------------------------------------------------------
result = run_scattnlay_spectrum(
    stack=stack,
    wavelengths_m=wavelengths_m,
    theta_rad=theta_rad,
    n_medium=medium,
)

# ------------------------------------------------------------
# 5) Backward collection with NA (reflectance-like proxy)
# ------------------------------------------------------------
backward_na = integrate_collection_na(
    result,
    collection_na=0.8,
    direction="backward",
)

# Choose the spectrum that should feed the colour conversion
# Option A: backward collected cross-section [m^2]
spectrum_raw = backward_na.c_collected_m2.copy()

# Option B: backward collected fraction [-]
# spectrum_raw = backward_na.fraction_collected.copy()

# ------------------------------------------------------------
# 6) Compute colour properties
# ------------------------------------------------------------
color_props = compute_color_properties(
    wavelength_nm=wavelengths_nm,
    input_spec=spectrum_raw,
    normalize_input=True,     # normalize internally
    normalization="max",      # or "sum"
)

print("=== Colour properties ===")
print("CIELAB:", color_props["CIELAB"])
print("Performance:", color_props["Performance in Rosch-MacAdam Solid"])
print("XYZ:", color_props["XYZ"])
print("xyY:", color_props["xyY"])
print("sRGB:", color_props["sRGB"])
print("HSV:", color_props["HSV"])
print("Hex:", color_props["Hex"])
print("Warnings:", color_props["Warnings"])
print("Input Spectrum Info:", color_props["Input Spectrum Info"])
# %%
import numpy as np
import importlib

import materials
import geometry
import solver
import integration
import colour_adapter

importlib.reload(materials)
importlib.reload(geometry)
importlib.reload(solver)
importlib.reload(integration)
importlib.reload(colour_adapter)

from materials import MaterialFileSpec, load_materials
from geometry import build_bragg_onion_from_peak_wavelength, resolve_layer_stack
from solver import run_scattnlay_spectrum
from integration import integrate_collection_na
from colour_adapter import compute_colour_from_integrated_scattering

# ------------------------------------------------------------
# 1) Define wavelength and angle grids
# ------------------------------------------------------------
wavelengths_nm = np.arange(400, 701, 1)
wavelengths_m = wavelengths_nm * 1e-9

theta_deg = np.linspace(0, 180, 361)
theta_rad = np.deg2rad(theta_deg)

# ------------------------------------------------------------
# 2) Load materials
# ------------------------------------------------------------
file_PS   = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\PS(hPS)\x=0.0.txt"
file_P2VP = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\P2VP(PDP)\x=0.0.txt"
file_H2O  = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\Water_HaleQuerry.txt"

material_specs = {
    "PS": MaterialFileSpec(
        name="PS",
        path=file_PS,
        wavelength_unit="nm",
        skiprows=2,
        names=["Wavelength", "RefractiveIndex", "k"],
        n_column="RefractiveIndex",
        k_column="k",
        extrapolation="extrapolate",
    ),
    "P2VP": MaterialFileSpec(
        name="P2VP",
        path=file_P2VP,
        wavelength_unit="nm",
        skiprows=2,
        names=["Wavelength", "RefractiveIndex", "k"],
        n_column="RefractiveIndex",
        k_column="k",
        extrapolation="extrapolate",
    ),
    "H2O": MaterialFileSpec(
        name="H2O",
        path=file_H2O,
        wavelength_unit="um",
        skiprows=0,
        names=["Wavelength", "RefractiveIndex"],
        n_column="RefractiveIndex",
        k_column=None,
        extrapolation="extrapolate",
    ),
}

loaded = load_materials(material_specs)

mat_A = loaded["PS"]
mat_B = loaded["P2VP"]
medium = loaded["H2O"]

# ------------------------------------------------------------
# 3) Build geometry and resolve stack
# ------------------------------------------------------------
geom = build_bragg_onion_from_peak_wavelength(
    material_a=mat_A,
    material_b=mat_B,
    peak_wavelength_m=550e-9,
    outer_layer="A",
    n_layers=9,
    core_thickness_factor=0.5,
)

stack = resolve_layer_stack(
    geometry=geom,
    material_a=mat_A,
    material_b=mat_B,
)

# ------------------------------------------------------------
# 4) Run scattering simulation
# ------------------------------------------------------------
result = run_scattnlay_spectrum(
    stack=stack,
    wavelengths_m=wavelengths_m,
    theta_rad=theta_rad,
    n_medium=medium,
)

# ------------------------------------------------------------
# 5) Backward collection with NA (reflectance-like proxy)
# ------------------------------------------------------------
backward_na = integrate_collection_na(
    result,
    collection_na=0.8,
    direction="backward",
)

# ------------------------------------------------------------
# 6) Compute colour directly from integrated scattering
# ------------------------------------------------------------
colour_result = compute_colour_from_integrated_scattering(
    backward_na,
    quantity="c_collected_m2",      # or "fraction_collected"
    wavelength_min_nm=400.0,
    wavelength_max_nm=700.0,
    normalize_input=True,
    normalization="max",
)

print("=== Colour result from helper ===")
print("Quantity used:", colour_result.quantity)
print("Used wavelength range [nm]:", colour_result.wavelengths_nm[[0, -1]])
print("Used spectrum min/max:", np.min(colour_result.spectrum_used), np.max(colour_result.spectrum_used))

props = colour_result.color_properties
print("CIELAB:", props["CIELAB"])
print("Performance:", props["Performance in Rosch-MacAdam Solid"])
print("XYZ:", props["XYZ"])
print("xyY:", props["xyY"])
print("sRGB:", props["sRGB"])
print("HSV:", props["HSV"])
print("Hex:", props["Hex"])
print("Warnings:", props["Warnings"])
# %%
import importlib
import colour_solid_plotting

importlib.reload(colour_solid_plotting)

from colour_solid_plotting import plot_colour_in_rosch_macadam_solid

fig = plot_colour_in_rosch_macadam_solid(
    colour_result.color_properties,
    show_max_chroma=True,
    title="Back-collected colour in Rosch–MacAdam solid",
    solid_max_points=50000,
    solid_marker_size=1.5,
    solid_opacity=0.08,
    point_name="Bragg onion colour",
    point_size=10,
)

fig.show()

# %%
import numpy as np
import importlib

import materials
import sweep

importlib.reload(materials)
importlib.reload(sweep)

from materials import MaterialFileSpec, load_materials
from sweep import (
    run_bragg_onion_sweep,
    plot_sweep_metric,
    plot_sweep_heatmap,
    plot_sweep_colour_strip,
)

# ------------------------------------------------------------
# 1) Define wavelength / angle grids
# ------------------------------------------------------------
wavelengths_nm = np.arange(400, 701, 1)
wavelengths_m = wavelengths_nm * 1e-9

theta_deg = np.linspace(0, 180, 361)
theta_rad = np.deg2rad(theta_deg)

# ------------------------------------------------------------
# 2) Load materials
# ------------------------------------------------------------
file_PS   = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\PS(hPS)\x=0.0.txt"
file_P2VP = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\P2VP(PDP)\x=0.0.txt"
file_H2O  = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\Water_HaleQuerry.txt"

material_specs = {
    "PS": MaterialFileSpec(
        name="PS",
        path=file_PS,
        wavelength_unit="nm",
        skiprows=2,
        names=["Wavelength", "RefractiveIndex", "k"],
        n_column="RefractiveIndex",
        k_column="k",
        extrapolation="extrapolate",
    ),
    "P2VP": MaterialFileSpec(
        name="P2VP",
        path=file_P2VP,
        wavelength_unit="nm",
        skiprows=2,
        names=["Wavelength", "RefractiveIndex", "k"],
        n_column="RefractiveIndex",
        k_column="k",
        extrapolation="extrapolate",
    ),
    "H2O": MaterialFileSpec(
        name="H2O",
        path=file_H2O,
        wavelength_unit="um",
        skiprows=0,
        names=["Wavelength", "RefractiveIndex"],
        n_column="RefractiveIndex",
        k_column=None,
        extrapolation="extrapolate",
    ),
}

loaded = load_materials(material_specs)

mat_A = loaded["PS"]
mat_B = loaded["P2VP"]
medium = loaded["H2O"]

# ------------------------------------------------------------
# 3) Define sweep grid
# ------------------------------------------------------------
parameter_grid = {
    "peak_wavelength_m": np.array([450e-9, 500e-9, 550e-9, 600e-9]),
    "n_layers": [7, 9, 11],
    "outer_layer": ["A", "B"],
    "core_thickness_factor": [0.5],
}

# ------------------------------------------------------------
# 4) Run sweep
# ------------------------------------------------------------
sweep_result = run_bragg_onion_sweep(
    geometry_mode="peak_wavelength",
    parameter_grid=parameter_grid,
    material_a=mat_A,
    material_b=mat_B,
    n_medium=medium,
    wavelengths_m=wavelengths_m,
    theta_rad=theta_rad,
    collection_na=0.8,
    collection_direction="backward",
    integration_quantity_for_colour="c_collected_m2",
    colour_wavelength_min_nm=400,
    colour_wavelength_max_nm=700,
    colour_normalize_input=True,
    colour_normalization="max",
    store_full_results=False,
    progress=True,
)

df = sweep_result.dataframe

df_plot = df.rename(columns={
    "Performance in Rosch-MacAdam Solid_eta_C": "eta_C",
    "Performance in Rosch-MacAdam Solid_eta_L": "eta_L",
    "Performance in Rosch-MacAdam Solid_eta_Y": "eta_Y",
    "Hex_hex": "hex",
    "CIELAB_L": "L",
    "CIELAB_a": "a",
    "CIELAB_b": "b",
    "CIELAB_C": "C",
    "CIELAB_hue_deg": "hue_deg",
})

print(df.head())
print(df.columns.tolist())
# %%
# Different plotting examples from previous sweep result
# 1) eta_C vs design wavelength, grouped by number of layers
import matplotlib.pyplot as plt
from sweep import plot_sweep_metric

# outer layer A
plot_sweep_metric(
    df_plot[df_plot["outer_layer"] == "A"],
    x="design_peak_wavelength_nm",
    y="eta_C",
    hue="n_layers",
    scale="linear",
)
plt.title("eta_C vs design wavelength (outer layer = A)")
plt.show()

# outer layer B
plot_sweep_metric(
    df_plot[df_plot["outer_layer"] == "B"],
    x="design_peak_wavelength_nm",
    y="eta_C",
    hue="n_layers",
    scale="linear",
)
plt.title("eta_C vs design wavelength (outer layer = B)")
plt.show()
# > Which desing wavlength gives the strongest chroma performance
# %%
# 2) Maximum collected signal vs design wavelength in dB
# Reflectance like measurement
plot_sweep_metric(
    df_plot,
    x="design_peak_wavelength_nm",
    y="Ccollected_max_m2",
    hue="outer_layer",
    scale="db",
)
plt.title("Max backward collected signal vs design wavelength")
plt.show()

# split by n_layers
for n_layers in sorted(df_plot["n_layers"].unique()):
    plot_sweep_metric(
        df_plot[df_plot["n_layers"] == n_layers],
        x="design_peak_wavelength_nm",
        y="Ccollected_max_m2",
        hue="outer_layer",
        scale="db",
    )
    plt.title(f"Max backward collected signal vs design wavelength (n_layers={n_layers})")
    plt.show()
# %%
# 3) Heatmap of eta_C vs wavelength and layer count
# screening plot
from sweep import plot_sweep_heatmap

plot_sweep_heatmap(
    df_plot[df_plot["outer_layer"] == "A"],
    x="design_peak_wavelength_nm",
    y="n_layers",
    value="eta_C",
    scale="linear",
)
plt.title("eta_C heatmap (outer layer = A)")
plt.show()
# for outer_layer = B
plot_sweep_heatmap(
    df_plot[df_plot["outer_layer"] == "B"],
    x="design_peak_wavelength_nm",
    y="n_layers",
    value="eta_C",
    scale="linear",
)
plt.title("eta_C heatmap (outer layer = B)")
plt.show()
# %%
# 4) Colour strip of the resulting colours
from sweep import plot_sweep_colour_strip

plot_sweep_colour_strip(
    df_plot[df_plot["outer_layer"] == "A"].sort_values(
        by=["design_peak_wavelength_nm", "n_layers"]
    ),
    x="design_peak_wavelength_nm",
    colour_hex_col="hex",
)
plt.title("Resulting colours (outer layer = A)")
plt.show()
# with labels
plot_sweep_colour_strip(
    df_plot[df_plot["outer_layer"] == "B"].sort_values(
        by=["design_peak_wavelength_nm", "n_layers"]
    ),
    x="design_peak_wavelength_nm",
    colour_hex_col="hex",
    label_col="n_layers",
)
plt.title("Resulting colours (outer layer = B, labels = n_layers)")
plt.show()
# %%
# 5) Scatter in CIELAB a*-b* plane
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6))

for _, row in df_plot.iterrows():
    ax.scatter(
        row["a"],
        row["b"],
        color=row["hex"],
        s=120,
        edgecolor="black",
    )

ax.set_xlabel("a*")
ax.set_ylabel("b*")
ax.set_title("Sweep results in CIELAB a*-b* plane")
ax.axhline(0, color="gray", linewidth=0.8)
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_aspect("equal")
plt.show()
# %%
# 6) Scatter in hue–chroma space
fig, ax = plt.subplots(figsize=(7, 4))

for _, row in df_plot.iterrows():
    ax.scatter(
        row["hue_deg"],
        row["C"],
        color=row["hex"],
        s=100,
        edgecolor="black",
    )

ax.set_xlabel("Hue angle [deg]")
ax.set_ylabel("Chroma C*")
ax.set_title("Sweep results in hue–chroma space")
plt.show()
# %%
# useful ranking snippet
df_plot.sort_values(
    by=["eta_C", "fraction_collected_max"],
    ascending=[False, False]
)[[
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
]].head(10)
# %%
# Sweep results as markers in CIELAB inside the Rosch–MacAdam colour solid

from sweep import make_plotting_aliases

df_plot = make_plotting_aliases(df)
import importlib
import colour_solid_plotting
import plotly.graph_objects as go

importlib.reload(colour_solid_plotting)

from colour_solid_plotting import (
    load_rosch_macadam_colour_solid,
    load_rosch_macadam_max_chroma,
    plot_rosch_macadam_colour_solid,
    add_cielab_point,
)

# ------------------------------------------------------------
# 1) Load and plot Rosch–MacAdam colour solid
# ------------------------------------------------------------
solid_df = load_rosch_macadam_colour_solid()
fig = plot_rosch_macadam_colour_solid(
    solid_df,
    title="Sweep results in Rosch–MacAdam colour solid",
    max_points=50000,
    marker_size=1.2,
    opacity=0.06,
)

# Optional: add max-chroma markers
try:
    max_chroma_df = load_rosch_macadam_max_chroma()
    # If your colour_solid_plotting module exposes a public helper later,
    # you can replace the internal call below.
    from colour_solid_plotting import _add_max_chroma_markers  # optional internal helper
    fig = _add_max_chroma_markers(fig, max_chroma_df, size=4.0)
except Exception as exc:
    print(f"Could not add max-chroma markers: {exc}")

# ------------------------------------------------------------
# 2) Overlay sweep results as coloured CIELAB markers
# ------------------------------------------------------------
for _, row in df_plot.iterrows():
    extra_hover = [
        f"design λ: {row['design_peak_wavelength_nm']:.1f} nm",
        f"n_layers: {int(row['n_layers'])}",
        f"outer_layer: {row['outer_layer']}",
        f"diameter: {row['diameter_nm']:.1f} nm",
        f"eta_C: {row['eta_C']:.3f}",
        f"eta_L: {row['eta_L']:.3f}",
        f"eta_Y: {row['eta_Y']:.3f}",
        f"hex: {row['hex']}",
    ]

    fig = add_cielab_point(
        fig,
        L=float(row["L"]),
        a=float(row["a"]),
        b=float(row["b"]),
        name=f"case {int(row['case_index'])}",
        color=str(row["hex"]),
        size=7.5,
        symbol="diamond",
        line_color="black",
        line_width=1.0,
        extra_hover_lines=extra_hover,
    )

# ------------------------------------------------------------
# 3) Improve camera / layout
# ------------------------------------------------------------
fig.update_layout(
    scene=dict(
        xaxis_title="a*",
        yaxis_title="b*",
        zaxis_title="L*",
        aspectmode="cube",
        camera=dict(
            eye=dict(x=1.8, y=1.8, z=1.2)
        ),
    ),
    legend=dict(itemsizing="constant"),
)

fig.show()
# %%
# marker size scaled by eta_C
import importlib
import colour_solid_plotting
import plotly.graph_objects as go

importlib.reload(colour_solid_plotting)

from colour_solid_plotting import (
    load_rosch_macadam_colour_solid,
    load_rosch_macadam_max_chroma,
    plot_rosch_macadam_colour_solid,
    add_cielab_point,
)

# ------------------------------------------------------------
# 1) Load and plot Rosch–MacAdam colour solid
# ------------------------------------------------------------
solid_df = load_rosch_macadam_colour_solid()
fig = plot_rosch_macadam_colour_solid(
    solid_df,
    title="Sweep results in Rosch–MacAdam colour solid",
    max_points=50000,
    marker_size=1.2,
    opacity=0.06,
)

# Optional: add max-chroma markers
try:
    max_chroma_df = load_rosch_macadam_max_chroma()
    # If your colour_solid_plotting module exposes a public helper later,
    # you can replace the internal call below.
    from colour_solid_plotting import _add_max_chroma_markers  # optional internal helper
    fig = _add_max_chroma_markers(fig, max_chroma_df, size=4.0)
except Exception as exc:
    print(f"Could not add max-chroma markers: {exc}")

# ------------------------------------------------------------
# 2) Overlay sweep results as coloured CIELAB markers
# ------------------------------------------------------------
for _, row in df_plot.iterrows():
    size = 5.0 + 10.0 * float(row["eta_C"])

    extra_hover = [
        f"design λ: {row['design_peak_wavelength_nm']:.1f} nm",
        f"n_layers: {int(row['n_layers'])}",
        f"outer_layer: {row['outer_layer']}",
        f"eta_C: {row['eta_C']:.3f}",
        f"eta_L: {row['eta_L']:.3f}",
        f"eta_Y: {row['eta_Y']:.3f}",
    ]

    fig = add_cielab_point(
        fig,
        L=float(row["L"]),
        a=float(row["a"]),
        b=float(row["b"]),
        name=f"case {int(row['case_index'])}",
        color=str(row["hex"]),
        size=size,
        symbol="diamond",
        line_color="black",
        line_width=1.0,
        extra_hover_lines=extra_hover,
    )

# ------------------------------------------------------------
# 3) Improve camera / layout
# ------------------------------------------------------------
fig.update_layout(
    scene=dict(
        xaxis_title="a*",
        yaxis_title="b*",
        zaxis_title="L*",
        aspectmode="cube",
        camera=dict(
            eye=dict(x=1.8, y=1.8, z=1.2)
        ),
    ),
    legend=dict(itemsizing="constant"),
)

fig.show()
# %%
# Field plotting testing
import importlib
import matplotlib.pyplot as plt
import fields

importlib.reload(fields)

from fields import compute_field_map, plot_field_magnitude, plot_poynting_streamlines

field_result = compute_field_map(
    stack=stack,
    wavelength_m=550e-9,
    n_medium=medium,
    plane="xz",
    npts=301,
    extent_outer_radius_factor=2.2,
)

print("Computed field map at wavelength [nm]:", field_result.wavelength_m * 1e9)
print("E_total shape:", field_result.e_total.shape)
print("H_total shape:", field_result.h_total.shape)
print("S_total shape:", field_result.s_total.shape)

# ------------------------------------------------------------
# 1) Total |E|
# ------------------------------------------------------------
plot_field_magnitude(
    field_result,
    quantity="E",
    field_kind="total",
    scale="db",
    floor=1e-8,
    cmap="turbo",
    clip_percentile_high=99.5,
)
plt.show()

# ------------------------------------------------------------
# 2) Scattered-like |E|
# ------------------------------------------------------------
plot_field_magnitude(
    field_result,
    quantity="E",
    field_kind="scattered",
    scale="db",
    floor=1e-8,
    cmap="turbo",
    clip_percentile_high=99.5,
)
plt.show()

# ------------------------------------------------------------
# 3) Total Poynting streamlines over |S_total|
# ------------------------------------------------------------
plot_poynting_streamlines(
    field_result,
    flow_kind="total",
    background_quantity="S",
    background_kind="total",
    background_scale="db",
    background_cmap="inferno",
    streamline_density=1.1,
    streamline_color="white",
    normalize_vectors=False,
    min_speed_fraction=0.01,
)
plt.show()

# ------------------------------------------------------------
# 4) Scattered-like Poynting streamlines over |S_sca_like|
# ------------------------------------------------------------
plot_poynting_streamlines(
    field_result,
    flow_kind="scattered",
    background_quantity="S",
    background_kind="scattered",
    background_scale="db",
    background_cmap="inferno",
    streamline_density=1.0,
    streamline_color="white",
    normalize_vectors=False,
    min_speed_fraction=0.02,
)
plt.show()
# %%
# 2nd field plot test
import numpy as np
import importlib
import matplotlib.pyplot as plt

import materials
import geometry
import fields

importlib.reload(materials)
importlib.reload(geometry)
importlib.reload(fields)

from materials import MaterialFileSpec, load_materials
from geometry import build_bragg_onion_from_peak_wavelength, resolve_layer_stack
from fields import compute_field_map, plot_field_magnitude, plot_poynting_streamlines

# ------------------------------------------------------------
# 1) Load materials
# ------------------------------------------------------------
file_PS   = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\PS(hPS)\x=0.0.txt"
file_P2VP = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\P2VP(PDP)\x=0.0.txt"
file_H2O  = r"C:\Users\SchwarzN\OneDrive - Université de Fribourg\Institution\P1_BraggSpericalPigments\Simulation\RefractiveIndices\Water_HaleQuerry.txt"

material_specs = {
    "PS": MaterialFileSpec(
        name="PS",
        path=file_PS,
        wavelength_unit="nm",
        skiprows=2,
        names=["Wavelength", "RefractiveIndex", "k"],
        n_column="RefractiveIndex",
        k_column="k",
        extrapolation="extrapolate",
    ),
    "P2VP": MaterialFileSpec(
        name="P2VP",
        path=file_P2VP,
        wavelength_unit="nm",
        skiprows=2,
        names=["Wavelength", "RefractiveIndex", "k"],
        n_column="RefractiveIndex",
        k_column="k",
        extrapolation="extrapolate",
    ),
    "H2O": MaterialFileSpec(
        name="H2O",
        path=file_H2O,
        wavelength_unit="um",
        skiprows=0,
        names=["Wavelength", "RefractiveIndex"],
        n_column="RefractiveIndex",
        k_column=None,
        extrapolation="extrapolate",
    ),
}

loaded = load_materials(material_specs)

mat_A = loaded["PS"]
mat_B = loaded["P2VP"]
medium = loaded["H2O"]

# ------------------------------------------------------------
# 2) Build one geometry and resolve the layer stack
# ------------------------------------------------------------
geom = build_bragg_onion_from_peak_wavelength(
    material_a=mat_A,
    material_b=mat_B,
    peak_wavelength_m=550e-9,
    outer_layer="B",
    n_layers=10,
    core_thickness_factor=0.5,
)

stack = resolve_layer_stack(
    geometry=geom,
    material_a=mat_A,
    material_b=mat_B,
)

print("Layer labels:", stack.layer_labels)
print("Radii [nm]:", stack.radii_m * 1e9)

# ------------------------------------------------------------
# 3) Compute field map
# ------------------------------------------------------------
field_result = compute_field_map(
    stack=stack,
    wavelength_m=550e-9,
    n_medium=medium,
    plane="xz",
    npts=301,
    extent_outer_radius_factor=2.2,
)

print("Computed field map at wavelength [nm]:", field_result.wavelength_m * 1e9)
print("E_total shape:", field_result.e_total.shape)
print("H_total shape:", field_result.h_total.shape)
print("S_total shape:", field_result.s_total.shape)

# ------------------------------------------------------------
# 4) Total |E| map (useful for hotspots)
# ------------------------------------------------------------
plot_field_magnitude(
    field_result,
    quantity="E",
    field_kind="total",
    scale="db",
    floor=1e-8,
    cmap="turbo",
    clip_percentile_high=99.5,
)
plt.show()

# ------------------------------------------------------------
# 5) Scattered-like |E| map (often more interpretable)
# ------------------------------------------------------------
plot_field_magnitude(
    field_result,
    quantity="E",
    field_kind="scattered",
    scale="db",
    floor=1e-8,
    cmap="turbo",
    clip_percentile_high=99.5,
)
plt.show()

# ------------------------------------------------------------
# 6) TOTAL energy flow streamlines
#    This is the safest plot to show streamlines clearly.
# ------------------------------------------------------------
plot_poynting_streamlines(
    field_result,
    flow_kind="total",
    background_quantity="S",
    background_kind="total",
    background_scale="db",
    background_floor=1e-12,
    background_cmap="inferno",
    background_clip_percentile_high=99.5,
    streamline_density=1.0,
    streamline_color="white",
    streamline_linewidth=0.9,
    normalize_vectors=True,      # helps visibility
    min_speed_fraction=0.0,      # do NOT mask weak vectors
    show_boundaries=True,
)
plt.show()

# ------------------------------------------------------------
# 7) SCATTERED-LIKE energy flow streamlines
#    More subtle than total flow, but now should also show.
# ------------------------------------------------------------
plot_poynting_streamlines(
    field_result,
    flow_kind="scattered",
    background_quantity="S",
    background_kind="scattered",
    background_scale="db",
    background_floor=1e-14,
    background_cmap="inferno",
    background_clip_percentile_high=99.5,
    streamline_density=2.0,
    streamline_color="cyan",
    streamline_linewidth=0.9,
    normalize_vectors=True,      # helps visibility for weak scattered flow
    min_speed_fraction=0.0,      # do NOT mask weak vectors
    show_boundaries=True,
)
plt.show()
# %%
# Field line plotting test update 3
import importlib
import matplotlib.pyplot as plt
import fields

importlib.reload(fields)

from fields import (
    compute_field_map,
    plot_field_magnitude,
    plot_poynting_streamlines,
    make_line_seeds,
)

field_result = compute_field_map(
    stack=stack,
    wavelength_m=550e-9,
    n_medium=medium,
    plane="xz",
    npts=301,
    extent_outer_radius_factor=2.2,
)

print("Computed field map at wavelength [nm]:", field_result.wavelength_m * 1e9)
print("E_total shape:", field_result.e_total.shape)
print("H_total shape:", field_result.h_total.shape)
print("S_total shape:", field_result.s_total.shape)

# ------------------------------------------------------------
# 1) Total |E|
# ------------------------------------------------------------
plot_field_magnitude(
    field_result,
    quantity="E",
    kind="total",
    scale="db",
    floor=1e-8,
    cmap="turbo",
    clip_percentile_high=99.5,
)
plt.show()

# ------------------------------------------------------------
# 2) Scattered-like |E|
# ------------------------------------------------------------
plot_field_magnitude(
    field_result,
    quantity="E",
    kind="scattered",
    scale="db",
    floor=1e-8,
    cmap="turbo",
    clip_percentile_high=99.5,
)
plt.show()

# ------------------------------------------------------------
# 3) Total flow over |S_total|
# ------------------------------------------------------------
plot_poynting_streamlines(
    field_result,
    flow_kind="total",
    background_quantity="S",
    background_kind="total",
    background_scale="db",
    background_floor=1e-12,
    background_cmap="inferno",
    background_clip_percentile_high=99.5,
    streamline_density=1.0,
    streamline_color="white",
    streamline_linewidth=0.9,
    normalize_vectors=True,
    min_speed_fraction=0.0,
    mask_inside_sphere=False,
)
plt.show()

# ------------------------------------------------------------
# 4) Scattered-like flow over |S_scattered|
# ------------------------------------------------------------
plot_poynting_streamlines(
    field_result,
    flow_kind="scattered",
    background_quantity="S",
    background_kind="scattered",
    background_scale="db",
    background_floor=1e-14,
    background_cmap="inferno",
    background_clip_percentile_high=99.5,
    streamline_density=1.0,
    streamline_color="cyan",
    streamline_linewidth=0.9,
    normalize_vectors=True,
    min_speed_fraction=0.0,
    mask_inside_sphere=False,
)
plt.show()

# ------------------------------------------------------------
# 5) Delta flow over |S_delta|
#    Often the most intuitive perturbation-flow visualization
# ------------------------------------------------------------
plot_poynting_streamlines(
    field_result,
    flow_kind="delta",
    background_quantity="S",
    background_kind="delta",
    background_scale="db",
    background_floor=1e-14,
    background_cmap="inferno",
    background_clip_percentile_high=99.5,
    streamline_density=1.0,
    streamline_color="lime",
    streamline_linewidth=0.9,
    normalize_vectors=True,
    min_speed_fraction=0.0,
    mask_inside_sphere=False,
)
plt.show()

# ------------------------------------------------------------
# 6) Mask interior + manually seed flow lines from below
# ------------------------------------------------------------
seeds = make_line_seeds(
    start_nm=(-1400, -1500),
    end_nm=(1400, -1500),
    n_seeds=25,
)

plot_poynting_streamlines(
    field_result,
    flow_kind="delta",
    background_quantity="S",
    background_kind="delta",
    background_scale="db",
    background_floor=1e-14,
    background_cmap="inferno",
    background_clip_percentile_high=99.5,
    streamline_density=1.0,
    streamline_color="white",
    streamline_linewidth=0.9,
    normalize_vectors=True,
    min_speed_fraction=0.0,
    mask_inside_sphere=True,
    start_points_nm=seeds,
)
plt.show()