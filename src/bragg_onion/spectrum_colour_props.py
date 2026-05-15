from __future__ import annotations

# Spectrum_ColorProps.py
#
# Convert a spectrum into colour properties using colour-science.
#
# Key features:
# - convert spectrum -> XYZ, xyY, CIELAB, sRGB, HSV, Hex
# - compute Rosch-MacAdam-based colour performance metrics:
#   eta_C, eta_L, eta_Y
# - optional normalization of the input spectrum before colour conversion
# - handling of out-of-gamut sRGB values
#
# Notes:
# - wavelength input is expected in nanometers
# - input spectrum should usually be non-negative
# - if normalize_input=False, the caller is responsible for the chosen scaling
# - this module expects the CSV file
#       rosch_macadam_max_chroma_per_hue_1deg.csv
#   to be located in the same directory as this module

from pathlib import Path
from typing import Dict, Any, Literal

import colour  # conda install colour-science
import numpy as np
import pandas as pd
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
NormalizationMode = Literal["max", "sum"]


# ============================================================
# Paths
# ============================================================

MODULE_DIR = Path(__file__).resolve().parent
ROSCH_MACADAM_CSV = MODULE_DIR / "rosch_macadam_max_chroma_per_hue_1deg.csv"


# ============================================================
# Helpers
# ============================================================

def _as_1d_float_array(values: float | list[float] | np.ndarray, name: str) -> FloatArray:
    """Convert scalar or iterable to a 1D float64 NumPy array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or a 1D array.")
    return arr.astype(np.float64, copy=False)


def _validate_same_shape(x: np.ndarray, y: np.ndarray, x_name: str, y_name: str) -> None:
    """Validate that two arrays have the same shape."""
    if x.shape != y.shape:
        raise ValueError(
            f"{x_name} and {y_name} must have the same shape, got {x.shape} and {y.shape}."
        )


def _normalize_spectrum(
    input_spec: FloatArray,
    normalize_input: bool,
    normalization: NormalizationMode,
) -> FloatArray:
    """
    Normalize input spectrum if requested.

    Parameters
    ----------
    input_spec :
        Input spectrum
    normalize_input :
        If True, normalize before colour conversion
    normalization :
        - "max": divide by max(input_spec)
        - "sum": divide by sum(input_spec)

    Returns
    -------
    np.ndarray
        Normalized or unchanged spectrum
    """
    spec = np.asarray(input_spec, dtype=float)

    if np.any(spec < 0):
        raise ValueError("input_spec must be non-negative for colour conversion.")

    if not normalize_input:
        return spec

    if normalization == "max":
        denom = np.max(spec)
    elif normalization == "sum":
        denom = np.sum(spec)
    else:
        raise ValueError(f"Unsupported normalization mode: {normalization!r}")

    if denom > 0:
        return spec / denom

    return np.zeros_like(spec, dtype=float)


def _clip_srgb(rgb: FloatArray) -> tuple[FloatArray, bool]:
    """
    Clip sRGB values into [0, 1].

    Returns
    -------
    rgb_clipped :
        Clipped RGB values
    was_clipped :
        True if clipping was necessary
    """
    rgb = np.asarray(rgb, dtype=float)
    clipped = np.clip(rgb, 0.0, 1.0)
    was_clipped = not np.allclose(rgb, clipped)
    return clipped.astype(np.float64), was_clipped


# ============================================================
# Rosch-MacAdam helper
# ============================================================

def hue_maxchroma_properties(target_hue: float) -> dict[str, float | str]:
    """
    Get the properties of the colour with maximum chroma at a given hue
    in the Rosch-MacAdam colour solid.

    Returns a dict with keys:
    - 'C'
    - 'L'
    - 'Y_rel'
    - 'a'
    - 'b'
    - 'hue_deg'
    - 'hex'
    """
    if not ROSCH_MACADAM_CSV.exists():
        raise FileNotFoundError(
            f"Could not find Rosch-MacAdam CSV file: {ROSCH_MACADAM_CSV}"
        )

    df_max_chroma = pd.read_csv(ROSCH_MACADAM_CSV)

    required_cols = {
        "hue_deg",
        "a_smooth",
        "b_smooth",
        "L_smooth",
        "hex_smooth",
    }
    missing = required_cols.difference(df_max_chroma.columns)
    if missing:
        raise ValueError(
            f"Rosch-MacAdam CSV is missing required columns: {sorted(missing)}"
        )

    diff = (df_max_chroma["hue_deg"] - target_hue).abs()
    ang_dist = np.minimum(diff, 360 - diff)
    row = df_max_chroma.loc[ang_dist.idxmin()]

    a = float(row["a_smooth"])
    b = float(row["b_smooth"])
    L = float(row["L_smooth"])
    hex_value = str(row["hex_smooth"])

    chroma = float(np.sqrt(a**2 + b**2))
    Y_rel = float(colour.Lab_to_XYZ(np.array([L, a, b]))[1] * 100.0)

    return {
        "C": chroma,
        "L": L,
        "Y_rel": Y_rel,
        "a": a,
        "b": b,
        "hue_deg": float(target_hue),
        "hex": hex_value,
    }


# ============================================================
# Spectrum and colour conversions
# ============================================================

def colour_spectrum(
    wavelength_nm: float | list[float] | np.ndarray,
    intensity: float | list[float] | np.ndarray = 1.0,
) -> colour.SpectralDistribution:
    """
    Generate a spectral distribution from wavelength and intensity arrays.

    Parameters
    ----------
    wavelength_nm :
        Wavelengths in nanometers, shape (N,)
    intensity :
        Intensity values, shape (N,)

    Returns
    -------
    colour.SpectralDistribution
    """
    wl = _as_1d_float_array(wavelength_nm, "wavelength_nm")
    inten = _as_1d_float_array(intensity, "intensity")
    _validate_same_shape(wl, inten, "wavelength_nm", "intensity")

    # Snap nearly-integer wavelength grids to exact integers in nm.
    wl = np.round(wl, 6)
    if np.allclose(wl, np.round(wl), atol=1e-6):
        wl = np.round(wl).astype(float)


    sd = colour.SpectralDistribution(dict(zip(wl.tolist(), inten.tolist())))
    return sd


def SpectrumToXYZ(
    spectrum: colour.SpectralDistribution,
    cmfs_name: str = "CIE 1931 2 Degree Standard Observer",
    illuminant_name: str = "D65",
) -> tuple[float, float, float]:
    """
    Convert a spectral distribution to CIE XYZ tristimulus values.

    Returns
    -------
    X, Y, Z
        XYZ tristimulus values in the range typically used by colour-science
    """
    cmfs = colour.MSDS_CMFS[cmfs_name]
    illuminant = colour.SDS_ILLUMINANTS[illuminant_name]
    XYZ = colour.sd_to_XYZ(spectrum, cmfs=cmfs, illuminant=illuminant)
    # Integration for arbitrary spectra grid
    # XYZ = colour.sd_to_XYZ(
    #     spectrum,
    #     cmfs=cmfs,
    #     illuminant=illuminant,
    #     method="Integration",
    # )
    return float(XYZ[0]), float(XYZ[1]), float(XYZ[2])


def XYZToCIELAB(
    X: float,
    Y: float,
    Z: float,
    illuminant_name: str = "D65",
) -> tuple[float, float, float]:
    """
    Convert CIE XYZ tristimulus values to CIELAB.

    Returns
    -------
    L*, a*, b*
    """
    illuminant = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][illuminant_name]
    Lab = colour.XYZ_to_Lab(np.array([X, Y, Z]), illuminant)
    return float(Lab[0]), float(Lab[1]), float(Lab[2])


def colour_performance(
    hue: float,
    chroma: float,
    lightness: float,
    Y_rel: float,
) -> tuple[float, float, float]:
    """
    Compute colour performance metrics in the Rosch-MacAdam colour solid.

    Returns
    -------
    eta_C, eta_L, eta_Y
    """
    hue_maxchroma_props = hue_maxchroma_properties(hue)

    # eta chroma or saturation as fraction of maximum chroma at this hue in Rosch-MacAdam solid
    eta_C = (
        chroma / float(hue_maxchroma_props["C"])
        if float(hue_maxchroma_props["C"]) > 0
        else 0.0
    )

    # # eta lightness as deviation score from maximum chroma lightness at this hue in Rosch-MacAdam solid.
    # # Sign is positive if lightness is above the maximum chroma lightness, negative if below.
    # target_L = float(hue_maxchroma_props["L"])
    # delta = lightness - target_L

    # delta_max = max(target_L, (100 - target_L))
    
    # eta_L = (1.0 -abs(delta) / delta_max if delta_max > 0 else 0.0)
    # eta_L_signed = eta_L if delta >= 0 else -eta_L

    
    # eta lightness as simple fraction of maximum chroma lightness at this hue in Rosch-MacAdam solid.
    eta_L = (
        lightness / float(hue_maxchroma_props["L"])
        if float(hue_maxchroma_props["L"]) > 0
        else 0.0
    )

    # # eta Y as deviation score from maximum chroma Y_rel at this hue in Rosch-MacAdam solid
    # # Sign is positive if Y_rel is above the maximum chroma Y_rel, negative if below.
    # target_Y_rel = float(hue_maxchroma_props["Y_rel"])
    # delta_Y = Y_rel - target_Y_rel

    # delta_Y_max = max(target_Y_rel, (100 - target_Y_rel))

    # eta_Y = (1.0 - abs(delta_Y) / delta_Y_max if delta_Y_max > 0 else 0.0)
    # eta_Y_signed = eta_Y if delta_Y >= 0 else -eta_Y

    # eta Y as simple fraction of maximum chroma Y_rel at this hue in Rosch-MacAdam solid.
    eta_Y = (
        Y_rel / float(hue_maxchroma_props["Y_rel"])
        if float(hue_maxchroma_props["Y_rel"]) > 0
        else 0.0
    )

    return float(eta_C), float(eta_L), float(eta_Y)


def srgb_to_hex(r: float, g: float, b: float) -> str:
    """
    Convert gamma-encoded sRGB values in [0, 1] to hex string '#RRGGBB'.
    """
    r8 = int(np.clip(r * 255.0, 0, 255))
    g8 = int(np.clip(g * 255.0, 0, 255))
    b8 = int(np.clip(b * 255.0, 0, 255))
    return f"#{r8:02X}{g8:02X}{b8:02X}"


# ============================================================
# Main API
# ============================================================

def compute_color_properties(
    wavelength_nm: np.ndarray,
    input_spec: np.ndarray,
    *,
    normalize_input: bool = False,
    normalization: NormalizationMode = "max",
    cmfs_name: str = "CIE 1931 2 Degree Standard Observer",
    illuminant_name: str = "D65",
) -> Dict[str, Dict[str, Any]]:
    """
    Compute colour properties from a spectral distribution.

    Parameters
    ----------
    wavelength_nm :
        Wavelengths in nanometers, shape (N,)
    input_spec :
        Spectral intensity values, shape (N,)
        Recommended to be non-negative.
        If normalize_input=False, the spectrum is used as provided.
    normalize_input :
        If True, normalize input_spec before colour conversion
    normalization :
        Normalization mode used if normalize_input=True:
        - "max": divide by maximum
        - "sum": divide by sum
    cmfs_name :
        Colour matching function set
    illuminant_name :
        Illuminant used for XYZ / Lab conversion

    Returns
    -------
    dict
        Dictionary containing:
        - CIELAB
        - Performance in Rosch-MacAdam Solid
        - XYZ
        - xyY
        - sRGB
        - HSV
        - Hex
        - Input Spectrum Info
        - Warnings
    """
    wl = _as_1d_float_array(wavelength_nm, "wavelength_nm")
    spec = _as_1d_float_array(input_spec, "input_spec")
    _validate_same_shape(wl, spec, "wavelength_nm", "input_spec")

    spec_used = _normalize_spectrum(
        input_spec=spec,
        normalize_input=normalize_input,
        normalization=normalization,
    )

    scale = 1e-2  # convert XYZ from [0..100] convention into [0..1] where needed

    sd = colour_spectrum(wl, spec_used)

    X, Y, Z = SpectrumToXYZ(
        sd,
        cmfs_name=cmfs_name,
        illuminant_name=illuminant_name,
    )

    xyz_scaled = np.array([X * scale, Y * scale, Z * scale], dtype=float)

    xy = colour.XYZ_to_xy(xyz_scaled)
    xyY = np.array([xy[0], xy[1], Y], dtype=float)
    # In CIE xyY:
    # - x and y are chromaticity coordinates in [0, 1]
    # - Y is kept on the same scale as the Y tristimulus value (here typically 0..100)

    rgb_raw = np.asarray(colour.XYZ_to_sRGB(xyz_scaled), dtype=float)
    rgb_clipped, was_clipped = _clip_srgb(rgb_raw)

    h, s, v = colour.RGB_to_HSV(rgb_clipped)
    hex_value = srgb_to_hex(*rgb_clipped)

    lightness, a, b_ = XYZToCIELAB(
        X * scale,
        Y * scale,
        Z * scale,
        illuminant_name=illuminant_name,
    )
    chroma = float(np.sqrt(a**2 + b_**2))
    hue_deg = float((np.degrees(np.arctan2(b_, a)) + 360.0) % 360.0)

    eta_C, eta_L, eta_Y = colour_performance(hue_deg, chroma, lightness, Y)

    warnings: list[str] = []
    if was_clipped:
        warnings.append(
            "Raw sRGB values were outside [0, 1] and were clipped before HSV/Hex conversion."
        )

    return {
        "CIELAB": {
            "L": float(lightness),
            "a": float(a),
            "b": float(b_),
            "C": float(chroma),
            "hue_deg": float(hue_deg),
        },
        "Performance in Rosch-MacAdam Solid": {
            "eta_C": float(eta_C),
            "eta_L": float(eta_L),
            "eta_Y": float(eta_Y),
        },
        "XYZ": {
            "X": float(X),
            "Y": float(Y),
            "Z": float(Z),
        },
        "xyY": {
            "x": float(xyY[0]),
            "y": float(xyY[1]),
            "Y": float(xyY[2]),
        },
        "sRGB": {
            "r": float(rgb_raw[0]),
            "g": float(rgb_raw[1]),
            "b": float(rgb_raw[2]),
            "r_clipped": float(rgb_clipped[0]),
            "g_clipped": float(rgb_clipped[1]),
            "b_clipped": float(rgb_clipped[2]),
            "was_clipped": bool(was_clipped),
        },
        "HSV": {
            "h": float(h),
            "s": float(s),
            "v": float(v),
        },
        "Hex": {
            "hex": hex_value,
        },
        "Input Spectrum Info": {
            "normalize_input": bool(normalize_input),
            "normalization": normalization if normalize_input else "none",
            "input_min": float(np.min(spec)),
            "input_max": float(np.max(spec)),
            "used_min": float(np.min(spec_used)),
            "used_max": float(np.max(spec_used)),
        },
        "Warnings": {
            "messages": warnings,
        },
    }