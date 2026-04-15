# mie_bragg_onion

**This project is currently under active research development.**  
APIs may change without notice.

Scattnlay-based Python tools for multilayer **Bragg onion spheres**:
wavelength-dependent materials, multilayer geometry generation, scattering simulation,
NA-based collection integration, colour analysis, parameter sweeps, and near-field / Poynting-flow visualization.

---

## Features

- **Material handling**
  - load dispersive refractive index data from files
  - interpolate `n(λ) + i k(λ)`
  - support wavelength units in nm / µm / m

- **Bragg onion geometry**
  - alternating A/B multilayer spheres
  - explicit thickness mode
  - quarter-wave design mode from peak wavelength
  - selectable outer layer
  - optional extra outer shell
  - optional extinction-coefficient modifications

- **Scattering** *(optional simulation stack)*
  - wrapper around `scattnlay`
  - differential scattering cross-sections
  - efficiencies and absolute cross-sections
  - angle-resolved scattering spectra

- **Collection / reflectance-like metrics** *(optional simulation stack)*
  - integrate scattering over collection numerical aperture (NA)
  - forward / backward collection
  - collected fraction and collected cross-section

- **Colour analysis**
  - XYZ, xyY, CIELAB, sRGB, HSV, Hex
  - Rosch–MacAdam colour-solid based performance metrics

- **Sweep / screening** *(optional simulation stack)*
  - scan design wavelength, number of layers, outer layer, and more
  - collect metrics into a tidy `pandas.DataFrame`
  - plot sweep trends and resulting colours

- **Near field** *(optional simulation stack)*
  - field maps in selected planes
  - total / scattered-like / delta-flow visualizations
  - Poynting streamlines

---

## Installation

### 1) Core package

Install the core package in editable mode from the repository root:

```bash
python -m pip install -e .
```

### 2) Core package without dependency resolution

If you want to install the core package without resolving dependencies, you can use:

```bash
python -m pip install -e . --no-deps
```

### 3) Optional simulation stack

The following modules depend on scattnlay:
- `solver.py`
- `field.py`
- `integration.py`
- `integration.py`
- `plotting.py`
- `colour_adapter.py`

Install the optional simulation dependencies with:

```bash
python -m pip install ".[scattnlay]"
```

### 4) Installing scattnlay directly

If you need to install the simulation dependency separately, try:
```bash
python -m pip install scattnlay
```

### 5) Windows note for scattnlay

On Windows, installing scattnlay may require:

- Microsoft C++ Build Tools
- a compatible MSVC / Python architecture
-`successful upstream build compatibility

A working installation may depend on:

- Python version
- 64-bit vs 32-bit architecture
- compiler / toolchain configuration

If the optional dependency cannot be built on your system, the core package
can still be installed and used independently.

## Quick start

### Core imports
```python
from bragg_onion import materials, geometry, spectrum_colour_props, colour_solid_plotting
```

### Example core workflow
```python
from bragg_onion.materials import ConstantDispersion
from bragg_onion.geometry import build_bragg_onion_from_peak_wavelength, resolve_layer_stack

mat_A = ConstantDispersion.from_nk("A", n=1.59, k=0.0)
mat_B = ConstantDispersion.from_nk("B", n=1.49, k=0.0)
medium = ConstantDispersion.from_nk("medium", n=1.33, k=0.0)

geom = build_bragg_onion_from_peak_wavelength(
    material_a=mat_A,
    material_b=mat_B,
    peak_wavelength_m=550e-9,
    outer_layer="A",
    n_layers=7,
    core_thickness_factor=0.5,
)

stack = resolve_layer_stack(
    geometry=geom,
    material_a=mat_A,
    material_b=mat_B,
)
```

## Smoke tests

### Core smoke test

A core smoke test script is provided in:
```bash
examples/smoke_test_core.py
```
Run it with:
```bash
python examples/smoke_test_core.py
```

### Sweep / analysis example

A package-based analysis script is provided in:
```bash
examples/analyse_bragg_onions.py
```
This script:
- loads dispersive materials
- runs a Bragg onion parameter sweep
- saves CSV results
- generates screening figures

Run it with:
```bash
python examples/analyse_bragg_onions.py
```
Before running it, edit the material-file paths at the top of the script so they match your local machine.

## Build the package

To build source and wheel distributions:
```bash
python -m pip install build
python -m build
```
This will create files in the dist/ directory, typically:
```bash
dist/
  bragg_onion-0.1.0.tar.gz
  bragg_onion-0.1.0-py3-none-any.whl
```

## Package structure
```bash
src/bragg_onion/
    __init__.py
    materials.py
    geometry.py
    solver.py
    integration.py
    plotting.py
    spectrum_colour_props.py
    colour_adapter.py
    colour_solid_plotting.py
    sweep.py
    fields.py
examples/
    smoke_test_core.py
    analyse_bragg_onions.py
```
## Recommended workflow

A typical workflow is:

1. Load dispersive materials
2. Build a Bragg onion geometry
3. Resolve the layer stack
4. Compute scattering (optional simulation stack)
5. Integrate over collection NA (optional simulation stack)
6. Compute colour properties
7. Inspect sweep results (optional simulation stack)
8. Inspect near fields (optional simulation stack)

## Project status

This repository currently provides:

- a validated **core package**
- an **optional simulation stack** that depends on scattnlay

The optional simulation stack is actively being stabilized across platforms.

## License

This project is licensed under the terms of the **GNU General Public License v3.0 or later**.
See the LICENSE file for details.

## Author

Niklas Richard Saverio Rocca Schwarz