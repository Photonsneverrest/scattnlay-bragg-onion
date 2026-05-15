from __future__ import annotations

import numpy as np
import pandas as pd
import colour
from matplotlib.colors import to_hex


def add_display_normalized_colour_columns(
    df: pd.DataFrame,
    *,
    inplace: bool = False,
    illuminant_name: str = "D65",
    observer_name: str = "CIE 1931 2 Degree Standard Observer",
) -> pd.DataFrame:
    """
    Add display-normalized colour columns to a dataframe.

    This is intended for object-colour-style visualisation, not for replacing
    the raw signal-based colour metrics.

    Required input columns:
      - X, Y, Z
      - Input Spectrum Info_used_max

    Returns the dataframe with added columns:
      - L_display, a_display, b_display
      - hex_display
      - X_display, Y_display, Z_display
    """
    required_cols = ["X", "Y", "Z", "Input Spectrum Info_used_max"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for display-normalized colours: {missing}")

    out = df if inplace else df.copy()

    illuminant_xy = colour.CCS_ILLUMINANTS[observer_name][illuminant_name]

    spec_max = out["Input Spectrum Info_used_max"].to_numpy(dtype=float)
    spec_max_safe = np.where(spec_max > 0, spec_max, 1.0)

    # Stored X,Y,Z in your sweep are in approximately 0..100 convention.
    # Normalize by used spectrum max, then convert to 0..1 XYZ.
    xyz_display_01 = (
        out[["X", "Y", "Z"]].to_numpy(dtype=float) / spec_max_safe[:, None]
    ) * 1e-2

    xyz_display_01 = np.clip(xyz_display_01, 0.0, None)

    lab_display = colour.XYZ_to_Lab(xyz_display_01, illuminant=illuminant_xy)

    out["L_display"] = lab_display[:, 0]
    out["a_display"] = lab_display[:, 1]
    out["b_display"] = lab_display[:, 2]

    rgb_display = colour.XYZ_to_sRGB(xyz_display_01)
    rgb_display = np.clip(rgb_display, 0.0, 1.0)

    out["hex_display"] = [to_hex(rgb) for rgb in rgb_display]

    out["X_display"] = xyz_display_01[:, 0]
    out["Y_display"] = xyz_display_01[:, 1]
    out["Z_display"] = xyz_display_01[:, 2]

    return out