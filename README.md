# mie_bragg_onion

**This project is currently under active research development.**
APIs may change without notice.

Scattnlay-based simulation tools for multilayer **Bragg onion spheres**:
wavelength-dependent materials, multilayer geometry generation, scattering simulation,
NA-based collection integration, colour analysis, parameter sweeps, and near-field / Poynting-flow visualization.

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

- **Scattering**
  - wrapper around `scattnlay`
  - differential scattering cross-sections
  - efficiencies and absolute cross-sections
  - angle-resolved scattering spectra

- **Collection / reflectance-like metrics**
  - integrate scattering over collection numerical aperture (NA)
  - forward / backward collection
  - collected fraction and collected cross-section

- **Colour analysis**
  - XYZ, xyY, CIELAB, sRGB, HSV, Hex
  - Rosch–MacAdam colour-solid based performance metrics

- **Sweep / screening**
  - scan design wavelength, number of layers, outer layer, and more
  - collect metrics into a tidy `pandas.DataFrame`
  - plot sweep trends and resulting colours

- **Near field**
  - field maps in selected planes
  - total / scattered-like / delta-flow visualizations
  - Poynting streamlines

## Repository structure

```text
src/bragg_onion/
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

## License

This project is licensed under the terms of the GNU General Public License v3.0.
See the `LICENSE` file for details.

## Author

Niklas Rocca Schwarz
