from __future__ import annotations

# integration.py
#
# Integrate angle-resolved scattering over solid angle.
#
# Key features:
# - convert collection NA into a maximum collection angle
# - integrate differential scattering over a cone or a theta range
# - support forward and backward collection
# - return collected scattering cross-section [m^2]
# - return collected fraction relative to total scattering [-]
# - return geometric-area-normalized collected scattering [-]
#
# Conventions:
# - theta is the polar scattering angle in radians
# - theta = 0   : forward direction
# - theta = π   : backward direction
# - assumes azimuthal symmetry around the incident axis
# - integration uses:
#       dΩ = 2π sin(theta) dtheta

from dataclasses import dataclass
from typing import Iterable, Literal, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .solver import ScatteringResult


# ============================================================
# Type aliases
# ============================================================

FloatArray = NDArray[np.float64]
CollectionDirection = Literal["forward", "backward"]

__all__ = [
    "IntegratedScatteringResult",
    "integrate_theta_range",
    "integrate_collection_na",
]


# ============================================================
# Helpers
# ============================================================

def geometric_cross_section_from_radii_m(radii_m) -> float:
    """
    Geometric cross-section πR² from the outer radius of the sphere.
    """
    r = float(np.asarray(radii_m, dtype=float)[-1])
    return float(np.pi * r**2)


def _as_1d_float_array(values: float | Iterable[float] | np.ndarray, name: str) -> FloatArray:
    """Convert scalar or iterable to a 1D float64 NumPy array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim != 1:
        raise ValueError(f"{name} must be scalar or 1D.")
    return arr.astype(np.float64, copy=False)

def _solid_angle_of_cone(theta_max: FloatArray) -> FloatArray:
    """
    Solid angle of a cone with half-angle theta_max.

    Ω = 2π (1 - cos(theta_max))
    """
    return 2.0 * np.pi * (1.0 - np.cos(theta_max))


def _solid_angle_of_theta_band(theta_min: FloatArray, theta_max: FloatArray) -> FloatArray:
    """
    Solid angle of a polar band from theta_min to theta_max.

    Ω = 2π (cos(theta_min) - cos(theta_max))
    """
    return 2.0 * np.pi * (np.cos(theta_min) - np.cos(theta_max))


def _validate_theta_range(theta_min: FloatArray, theta_max: FloatArray) -> None:
    if np.any(theta_min < 0) or np.any(theta_max > np.pi):
        raise ValueError("Theta must satisfy 0 ≤ θ ≤ π.")
    if np.any(theta_max < theta_min):
        raise ValueError("theta_max must be ≥ theta_min.")


def na_to_theta_max(
    collection_na: float,
    n_medium: float | complex | FloatArray | NDArray[np.complex128],
) -> FloatArray:
    """
    Convert numerical aperture to collection half-angle in the surrounding medium.

    Uses:
        theta_max = arcsin(NA / Re(n_medium))

    Parameters
    ----------
    collection_na :
        Numerical aperture
    n_medium :
        Real/complex refractive index or wavelength-dependent array

    Returns
    -------
    np.ndarray
        Theta max in radians
    """
    na = float(collection_na)
    if na < 0:
        raise ValueError("collection_na must be non-negative.")
    
    n_med = np.real(np.asarray(n_medium, dtype=float))
    if np.any(n_med <= 0):
        raise ValueError("Real(n_medium) must be positive.")
    
    ratio = na / n_med
    if np.any(ratio > 1.0):
        raise ValueError("collection_na exceeds Re(n_medium).")
    return np.arcsin(ratio)


def _integrate_dcs(
    theta_rad: FloatArray,
    dcs_row: FloatArray,
    mask: NDArray[np.bool_],
) -> float:
    """
    Integrate one wavelength row of dσ/dΩ over the selected theta region.

    Uses:
        C_collected = ∫ (dσ/dΩ) dΩ
                    = ∫ (dσ/dΩ) 2π sin(theta) dtheta
    """
    theta_sel = theta_rad[mask]
    dcs_sel = dcs_row[mask]

    if theta_sel.size < 2:
        return 0.0
    
    integrand = dcs_sel * 2.0 * np.pi * np.sin(theta_sel)
    return float(np.trapz(integrand, theta_sel))


# ============================================================
# Dataclass
# ============================================================

@dataclass(frozen=True)
class IntegratedScatteringResult:
    """
    Integrated scattering results over a selected solid-angle region.

    Attributes
    ----------
    wavelengths_m :
        Wavelength grid [m], shape (n_wavelengths,)
    theta_min_rad, theta_max_rad :
        Polar angular limits of the integration region [rad]
    direction :
        "forward", "backward", or "custom"
    collection_na :
        Numerical aperture used for defining the region, or None
    c_collected_m2 :
        Collected scattering cross-section [m^2], shape (n_wavelengths,)
    fraction_collected :
        Collected fraction relative to total scattering cross-section [-],
        shape (n_wavelengths,)
    q_collected_geom :
        Collected scattering normalized by geometric cross-sectional area [-],
        shape (n_wavelengths,)
    solid_angle_sr :
        Solid angle of the integrated region [sr], shape (n_wavelengths,)
    geometric_cross_section_m2 :
        Geometric cross-sectional area of the particle [m^2]
    """
    wavelengths_m: FloatArray
    theta_min_rad: FloatArray
    theta_max_rad: FloatArray
    direction: str
    collection_na: float | None

    c_collected_m2: FloatArray
    fraction_collected: FloatArray
    q_collected_geom: FloatArray
    solid_angle_sr: FloatArray

    geometric_cross_section_m2: float


# ============================================================
# Public API
# ============================================================

def integrate_theta_range(
    result: ScatteringResult,
    theta_min_rad: float | Iterable[float] | np.ndarray,
    theta_max_rad: float | Iterable[float] | np.ndarray,
    *,
    direction: str = "custom",
    collection_na: float | None = None,
) -> IntegratedScatteringResult:
    """
    Integrate scattering over an explicit theta range.

    Parameters
    ----------
    result :
        Output from solver.run_scattnlay_spectrum(...)
    theta_min_rad, theta_max_rad :
        Lower and upper polar angle limits [rad]
        Can be scalars or arrays of shape (n_wavelengths,)
    direction :
        Label stored in the result metadata
    collection_na :
        Optional NA value stored in the result metadata

    Returns
    -------
    IntegratedScatteringResult
    """
    wl = np.asarray(result.wavelengths_m, dtype=float)
    theta = np.asarray(result.theta_rad, dtype=float)
    dcs = np.asarray(result.dcs_m2_sr, dtype=float)

    theta_min = _as_1d_float_array(theta_min_rad, "theta_min_rad")
    theta_max = _as_1d_float_array(theta_max_rad, "theta_max_rad")

    if theta_min.size == 1:
        theta_min = np.full_like(wl, float(theta_min[0]), dtype=float)
    if theta_max.size == 1:
        theta_max = np.full_like(wl, float(theta_max[0]), dtype=float)

    if theta_min.shape != wl.shape or theta_max.shape != wl.shape:
        raise ValueError(
            "theta_min_rad and theta_max_rad must be scalars or arrays with shape (n_wavelengths,)."
        )

    _validate_theta_range(theta_min, theta_max)

    c_collected = np.empty_like(wl, dtype=float)

    for i in range(wl.size):
        mask = (theta >= theta_min[i]) & (theta <= theta_max[i])
        c_collected[i] = _integrate_dcs(theta, dcs[i], mask)

    csca = np.asarray(result.csca_m2, dtype=float)
    G = geometric_cross_section_from_radii_m(result.radii_m)

    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(csca > 0, c_collected / csca, np.nan)
        q_geom = c_collected / G

    solid_angle = _solid_angle_of_theta_band(theta_min, theta_max)

    return IntegratedScatteringResult(
        wavelengths_m=wl,
        theta_min_rad=theta_min,
        theta_max_rad=theta_max,
        direction=direction,
        collection_na=collection_na,
        c_collected_m2=c_collected,
        fraction_collected=fraction,
        q_collected_geom=q_geom,
        solid_angle_sr=solid_angle,
        geometric_cross_section_m2=G,
    )


def integrate_collection_na(
    result: ScatteringResult,
    collection_na: float,
    *,
    direction: CollectionDirection = "backward",
) -> IntegratedScatteringResult:
    """
    Integrate scattering collected by a numerical-aperture cone.

    Parameters
    ----------
    result :
        Output from solver.run_scattnlay_spectrum(...)
    collection_na :
        Collection numerical aperture
    direction :
        - "forward"  : cone centered at theta = 0
        - "backward" : cone centered at theta = π

    Returns
    -------
    IntegratedScatteringResult
    """
    theta_max = na_to_theta_max(collection_na, result.n_medium)

    if direction == "forward":
        theta_min = np.zeros_like(theta_max)
        theta_upper = theta_max
    elif direction == "backward":
        theta_min = np.pi - theta_max
        theta_upper = np.full_like(theta_max, np.pi)
    else:
        raise ValueError("direction must be 'forward' or 'backward'.")

    return integrate_theta_range(
        result=result,
        theta_min_rad=theta_min,
        theta_max_rad=theta_upper,
        direction=direction,
        collection_na=collection_na,
    )