from __future__ import annotations

# colour_adapter.py
#
# Small helper module to connect the scattering/integration pipeline
# with spectrum_colour_props.compute_color_properties().
#
# Key features:
# - extract a chosen spectrum from IntegratedScatteringResult
# - optionally restrict to a wavelength range (e.g. visible only)
# - pass the spectrum into compute_color_properties()
# - return both the raw/used spectrum and the computed colour properties
#
# Intended usage:
# - run solver.py
# - integrate over collection NA with integration.py
# - call compute_colour_from_integrated_scattering(...)
#
# Typical reflectance-like use case:
# - use backward NA-collected scattering
# - choose quantity="c_collected_m2" or "fraction_collected"

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .integration import IntegratedScatteringResult

from .spectrum_colour_props import compute_color_properties


FloatArray = NDArray[np.float64]
IntegratedQuantity = Literal[
    "c_collected_m2",
    "fraction_collected",
    "q_collected_geom",
]


__all__ = [
    "ColourComputationResult",
    "compute_colour_from_integrated_scattering",
]


# ============================================================
# Dataclass
# ============================================================

@dataclass(frozen=True)
class ColourComputationResult:
    """
    Container for colour computation input/output.

    Attributes
    ----------
    wavelengths_nm :
        Wavelength grid in nanometers used for colour conversion
    quantity :
        Integrated scattering quantity used as the spectral input
    spectrum_raw :
        Raw selected spectrum before any masking or normalization
    spectrum_used :
        Spectrum actually passed to compute_color_properties()
        (after wavelength masking, but before internal normalization inside
        spectrum_colour_props if normalize_input=True is used there)
    color_properties :
        Dictionary returned by compute_color_properties()
    """
    wavelengths_nm: FloatArray
    quantity: str
    spectrum_raw: FloatArray
    spectrum_used: FloatArray
    color_properties: dict[str, Any]


# ============================================================
# Helpers
# ============================================================

# def _as_1d_float_array(values: float | list[float] | np.ndarray, name: str) -> FloatArray:
#     """Convert scalar or iterable to a 1D float64 array."""
#     arr = np.asarray(values, dtype=float)
#     if arr.ndim == 0:
#         arr = arr.reshape(1)
#     elif arr.ndim != 1:
#         raise ValueError(f"{name} must be a scalar or a 1D array.")
#     return arr.astype(np.float64, copy=False)


# def extract_integrated_spectrum(
#     integrated_result: IntegratedScatteringResult,
#     *,
#     quantity: IntegratedQuantity = "c_collected_m2",
#     wavelength_min_nm: float | None = None,
#     wavelength_max_nm: float | None = None,
# ) -> tuple[FloatArray, FloatArray]:
#     """
#     Extract one spectral quantity from an IntegratedScatteringResult.

#     Parameters
#     ----------
#     integrated_result :
#         Output of integration.integrate_theta_range(...) or integrate_collection_na(...)
#     quantity :
#         Which spectrum to extract:
#         - "c_collected_m2"
#         - "fraction_collected"
#         - "c_collected_geom_norm"
#     wavelength_min_nm, wavelength_max_nm :
#         Optional wavelength window in nanometers

#     Returns
#     -------
#     wavelengths_nm :
#         Wavelength grid in nm
#     spectrum :
#         Selected spectrum
#     """
#     allowed = {"c_collected_m2", "fraction_collected", "c_collected_geom_norm"}
#     if quantity not in allowed:
#         raise ValueError(f"quantity must be one of {allowed}.")

#     if not hasattr(integrated_result, quantity):
#         raise ValueError(f"IntegratedScatteringResult has no attribute '{quantity}'.")

#     wavelengths_nm = np.asarray(integrated_result.wavelengths_m, dtype=float) * 1e9

#     # Snap floating-point converted wavelengths onto an exact nm grid if appropriate.
#     wavelengths_nm = np.round(wavelengths_nm, 6)
#     if np.allclose(wavelengths_nm, np.round(wavelengths_nm), atol=1e-6):
#         wavelengths_nm = np.round(wavelengths_nm).astype(float)
#     spectrum = np.asarray(getattr(integrated_result, quantity), dtype=float)

#     if wavelengths_nm.ndim != 1 or spectrum.ndim != 1:
#         raise ValueError("Integrated result wavelength and selected quantity must both be 1D.")
#     if wavelengths_nm.shape != spectrum.shape:
#         raise ValueError(
#             f"Wavelength grid and selected spectrum must have the same shape, "
#             f"got {wavelengths_nm.shape} and {spectrum.shape}."
#         )

#     mask = np.ones_like(wavelengths_nm, dtype=bool)
#     if wavelength_min_nm is not None:
#         mask &= wavelengths_nm >= float(wavelength_min_nm)
#     if wavelength_max_nm is not None:
#         mask &= wavelengths_nm <= float(wavelength_max_nm)

#     wavelengths_nm = wavelengths_nm[mask]
#     spectrum = spectrum[mask]

#     if wavelengths_nm.size == 0:
#         raise ValueError("No wavelengths remain after applying the requested wavelength range.")

#     return wavelengths_nm.astype(np.float64), spectrum.astype(np.float64)


# ============================================================
# Main API
# ============================================================

# def compute_colour_from_integrated_scattering(
#     integrated_result,
#     *,
#     quantity: str = "q_collected_geom",
#     wavelength_min_nm: float | None = None,
#     wavelength_max_nm: float | None = None,
#     normalize_input: bool = False,
#     normalization: Literal["max", "sum"] = "max",
#     cmfs_name: str = "CIE 1931 2 Degree Standard Observer",
#     illuminant_name: str = "D65",
# ) -> ColourComputationResult:
#     """
#     Compute colour properties directly from integrated scattering data.

#     Parameters
#     ----------
#     integrated_result :
#         Output from integration.integrate_theta_range(...) or integrate_collection_na(...)
#     quantity :
#         Which integrated quantity should be interpreted as the spectrum:
#         - "c_collected_m2"
#         - "fraction_collected"
#         - "c_collected_geom_norm"
#     wavelength_min_nm, wavelength_max_nm :
#         Optional wavelength limits in nm. Defaults to visible range 400-700 nm.
#     normalize_input :
#         Passed through to spectrum_colour_props.compute_color_properties()
#     normalization :
#         Passed through to spectrum_colour_props.compute_color_properties()
#     cmfs_name :
#         Passed through to spectrum_colour_props.compute_color_properties()
#     illuminant_name :
#         Passed through to spectrum_colour_props.compute_color_properties()

#     Returns
#     -------
#     ColourComputationResult
#     """
#     wavelengths_nm, spectrum = extract_integrated_spectrum(
#         integrated_result,
#         quantity=quantity,
#         wavelength_min_nm=wavelength_min_nm,
#         wavelength_max_nm=wavelength_max_nm,
#     )

#     color_props = compute_color_properties(
#         wavelength_nm=wavelengths_nm,
#         input_spec=spectrum,
#         normalize_input=normalize_input,
#         normalization=normalization,
#         cmfs_name=cmfs_name,
#         illuminant_name=illuminant_name,
#     )

#     return ColourComputationResult(
#         wavelengths_nm=wavelengths_nm,
#         quantity=quantity,
#         spectrum_raw=np.asarray(getattr(integrated_result, quantity), dtype=float),
#         spectrum_used=spectrum,
#         color_properties=color_props,
#     )

def compute_colour_from_integrated_scattering(
    integrated_result: IntegratedScatteringResult,
    *,
    quantity: IntegratedQuantity = "q_collected_geom",
    wavelength_min_nm: float | None = None,
    wavelength_max_nm: float | None = None,
    normalize_input: bool = False,
    normalization: Literal["max", "sum"] = "max",
    cmfs_name: str = "CIE 1931 2 Degree Standard Observer",
    illuminant_name: str = "D65",
) -> ColourComputationResult:
    """
    Compute colour properties directly from integrated scattering data.
    
    Parameters
    ----------
    integrated_result :
        Output from integration.integrate_theta_range(...) or integrate_collection_na(...)
    quantity :
        Which integrated quantity should be interpreted as the spectrum:
        - "c_collected_m2"
        - "fraction_collected"
        - "c_collected_geom_norm"
    wavelength_min_nm, wavelength_max_nm :
        Optional wavelength limits in nm. Defaults to visible range 400-700 nm.
    normalize_input :
        Passed through to spectrum_colour_props.compute_color_properties()
    normalization :
        Passed through to spectrum_colour_props.compute_color_properties()
    cmfs_name :
        Passed through to spectrum_colour_props.compute_color_properties()
    illuminant_name :
        Passed through to spectrum_colour_props.compute_color_properties()

    Returns
    -------
    ColourComputationResult
    """

    if not hasattr(integrated_result, quantity):
        raise ValueError(f"Integrated result has no attribute '{quantity}'.")

    wavelengths_nm = np.asarray(integrated_result.wavelengths_m) * 1e9
    spectrum = np.asarray(getattr(integrated_result, quantity), dtype=float)

    mask = np.ones_like(wavelengths_nm, dtype=bool)
    if wavelength_min_nm is not None:
        mask &= wavelengths_nm >= wavelength_min_nm
    if wavelength_max_nm is not None:
        mask &= wavelengths_nm <= wavelength_max_nm

    wavelengths_nm = wavelengths_nm[mask]
    spectrum_used = spectrum[mask]

    color_props = compute_color_properties(
        wavelength_nm=wavelengths_nm,
        input_spec=spectrum_used,
        normalize_input=normalize_input,
        normalization=normalization,
        cmfs_name=cmfs_name,
        illuminant_name=illuminant_name,
    )

    return ColourComputationResult(
        wavelengths_nm=wavelengths_nm,
        quantity=quantity,
        spectrum_raw=spectrum,
        spectrum_used=spectrum_used,
        color_properties=color_props,
    )