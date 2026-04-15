from typing_extensions import Dict
import colour # conda install colour-science 
import numpy as np
import pandas as pd

# Import colour solid surface in CIELAB coordinate system
from RoschMacAdam_cache import load_colour_solid_from_csv 
from RoschMacAdam_analysis import max_chroma_at_hue, Lstar_to_Yrel

def hue_maxchroma_properties(target_hue: float) -> dict:
    """Get the properties of the color with maximum chroma at a given hue in the Rosch-MacAdam color solid just like its lightness.
    Returns a dict with keys: 'C', 'L', 'Y_rel', 'a', 'b', 'hue_deg'.
    """
    # Load data from rosch_macadam_max_chroma_per_hue_1deg.csv
    df_max_chroma = pd.read_csv('rosch_macadam_max_chroma_per_hue_1deg.csv')
    # Get closest row to target_hue (hue deg resolution is 1 deg)
    diff = (df_max_chroma['hue_deg'] - target_hue).abs()
    ang_dist = np.minimum(diff, 360 - diff)
    row = df_max_chroma.loc[ang_dist.idxmin()]
    # row = df_max_chroma[np.isclose(df_max_chroma['hue_deg'], target_hue, atol=0.6)]
    if not row.empty:
        a = row['a_smooth'] # Range: [-100..100]
        b = row['b_smooth'] # Range: [-100..100]
        L = row['L_smooth'] # Range: [0..100]
        hex = row['hex_smooth']
    chroma = np.sqrt(a**2 + b**2)
    # Calculate Y_rel from L*
    Y_rel = colour.Lab_to_XYZ(np.array([L, a, b]))[1] * 100.0 # scale back to [0..100]
    return {
        'C': chroma,
        'L': L,
        'Y_rel': Y_rel,
        'a': a,
        'b': b,
        'hue_deg': target_hue,
        'hex': hex
    }

def colour_spectrum(wavelength_nm: float, intensity: float = 1.0) -> colour.SpectralDistribution:
    """Generate a spectral distribution for a given wavelength and intensity.
    wavelength_nm: Wavelength in nanometers shape = [N]
    intensity: Intensity value shape = [N] in range [0..1]
    Returns: colour.SpectralDistribution 
    https://colour.readthedocs.io/en/develop/generated/colour.SpectralDistribution.html
    """
    sd = colour.SpectralDistribution(
        dict(zip(wavelength_nm.tolist(), intensity.tolist()))
    )
    return sd

def SpectrumToXYZ(spectrum: colour.SpectralDistribution, cmfs_name='CIE 1931 2 Degree Standard Observer', illuminant_name='D65') -> tuple[float, float, float]:
    """Convert a spectral distribution to CIE XYZ tristimulus values.
    Returns: X, Y, Z
    """
    cmfs = colour.MSDS_CMFS[cmfs_name]
    illuminant = colour.SDS_ILLUMINANTS[illuminant_name]
    XYZ = colour.sd_to_XYZ(spectrum, cmfs=cmfs, illuminant=illuminant)
    return XYZ[0], XYZ[1], XYZ[2] # XYZ range is [0, 100]

def XYZToCIELAB(X: float, Y: float, Z: float, illuminant_name='D65') -> tuple[float, float, float]:
    """Convert CIE XYZ tristimulus values to CIELAB color space.
    Returns: L*, a*, b*
    """
    illuminant = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant_name]
    Lab = colour.XYZ_to_Lab(np.array([X, Y, Z]), illuminant)
    return Lab[0], Lab[1], Lab[2] # LAB range L*: [0..100], a*: [-100..100], b*: [-100..100]

def colour_performance(hue, chroma, lightness, Y_rel) -> tuple[float, float, float]:
    """Compute performance metrics of a color in the Rosch-MacAdam color solid.
    Returns: C_performance, L_performance, Y_performance
    """
    hue_maxchroma_props = hue_maxchroma_properties(hue)
    eta_C = chroma / hue_maxchroma_props['C'] if hue_maxchroma_props['C'] > 0 else 0
    eta_L = lightness / hue_maxchroma_props['L'] if hue_maxchroma_props['L'] > 0 else 0
    eta_Y = Y_rel / hue_maxchroma_props['Y_rel'] if hue_maxchroma_props['Y_rel'] > 0 else 0
    return eta_C, eta_L, eta_Y

def srgb_to_hex(r: float, g: float, b: float) -> str:
    """Convert gamma-encoded sRGB (0..1) to hex string '#RRGGBB'."""
    r8 = int(np.clip(r * 255.0, 0, 255))
    g8 = int(np.clip(g * 255.0, 0, 255))
    b8 = int(np.clip(b * 255.0, 0, 255))
    return f"#{r8:02X}{g8:02X}{b8:02X}"

def compute_color_properties(
    wavelength_nm: np.ndarray,
    input_spec: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Compute various color properties from a spectral distribution.
    wavelength_nm: Wavelengths in nanometers shape = [N]
    input_spec: Intensity values in range [0..1] shape = [N]
    Returns: A dictionary with color properties in different color spaces and performance metrics.
    """
    scale = 1e-2 # Scale factor to convert from [0..1] to [0..100] for colour-science functions
    sd = colour_spectrum(wavelength_nm, input_spec)
    X, Y, Z = SpectrumToXYZ(sd) # XYZ range is [0, 100]
    xy = colour.XYZ_to_xy(np.array([X*scale, Y*scale, Z*scale])) # Convert from CIE XYZ tristimulus values to CIE xy chromaticity coordinates.
    xyY = np.array([xy[0]/scale, xy[1]/scale, Y]) # CIE xyY with ranges [0, 100]
    r, g, b = colour.XYZ_to_sRGB(np.array([X*scale, Y*scale, Z*scale])) # RGB range is [0, 1]
    h, s, v = colour.RGB_to_HSV(np.array([r, g, b])) # HSV range is [0, 1]
    hex = srgb_to_hex(r, g, b)
    lightness, a, b_ = XYZToCIELAB(X*scale, Y*scale, Z*scale) # LAB range L*: [0..100], a*: [-100..100], b*: [-100..100]
    chroma = np.sqrt(a**2 + b_**2)
    hue_deg = (np.degrees(np.arctan2(b_, a)) + 360) % 360

    # Compute performance in Rosch-MacAdam color solid. Use CIELAB hue, chroma, lightness.
    eta_C, eta_L, eta_Y = colour_performance(hue_deg, chroma, lightness, Y)

    return {
        'CIELAB': {'L': lightness, 'a': a, 'b': b_, 'C': chroma, 'hue_deg': hue_deg},
        'Performance in Rosch-MacAdam Solid': {
            'eta_C': eta_C, # Chroma resp. saturation performance
            'eta_L': eta_L, # Lightness performance compared to max chroma at hue
            'eta_Y': eta_Y  # Relative luminance performance compared to max chroma at hue
        },
        'XYZ': {'X': X, 'Y': Y, 'Z': Z},
        'xyY': {'x': xyY[0], 'y': xyY[1], 'Y': xyY[2]},
        'sRGB': {'r': r, 'g': g, 'b': b},
        'HSV': {'h': h, 's': s, 'v': v},
        'Hex': {'hex': hex}
    }