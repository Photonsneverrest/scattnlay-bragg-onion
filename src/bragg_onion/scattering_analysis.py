from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.cm import ScalarMappable

# ============================================================
# Basic helpers
# ============================================================

def geometric_cross_section_m2(scattering_result) -> float:
    """
    Geometric cross-section πR² from the outer radius stored in scattering_result.radii_m.
    """
    r = float(np.asarray(scattering_result.radii_m, dtype=float)[-1])
    return float(np.pi * r**2)


def _cross_section_array(scattering_result, which: str) -> np.ndarray:
    """
    Get an absolute cross-section spectrum [m²] for 'ext', 'sca', or 'abs'.
    Uses c*_m2 if available, otherwise q* * geometric cross-section.
    """
    c_name = f"c{which}_m2"
    if hasattr(scattering_result, c_name):
        return np.asarray(getattr(scattering_result, c_name), dtype=float)

    q_name = f"q{which}"
    if hasattr(scattering_result, q_name):
        G = geometric_cross_section_m2(scattering_result)
        return np.asarray(getattr(scattering_result, q_name), dtype=float) * G

    raise AttributeError(
        f"Could not find either '{c_name}' or '{q_name}' on scattering_result."
    )


def to_db(values, floor: float = 1e-30) -> np.ndarray:
    """
    Convert a non-negative quantity to dB using 10 log10.
    """
    arr = np.asarray(values, dtype=float)
    return 10.0 * np.log10(np.maximum(arr, floor))


def nearest_wavelength_index(wavelengths_m, wavelength_nm: float) -> int:
    wl_nm = np.asarray(wavelengths_m, dtype=float) * 1e9
    return int(np.argmin(np.abs(wl_nm - wavelength_nm)))


# ============================================================
# Angular quantities
# ============================================================

def angular_quantity(
    scattering_result,
    *,
    normalization: str = "geom",
) -> np.ndarray:
    """
    Return angle-dependent quantity with shape (n_wavelengths, n_theta).

    Parameters
    ----------
    normalization :
        "none" : absolute differential scattering cross-section dσ/dΩ [m² sr⁻¹]
        "geom" : differential efficiency (dσ/dΩ) / (πR²) [sr⁻¹]
        "sca"  : phase function = (dσ/dΩ) / σsca [sr⁻¹]

    Notes
    -----
    If `phase_function_sr_inv` already exists on the scattering result and
    `normalization="sca"`, it is used directly.
    """
    dcs = np.asarray(scattering_result.dcs_m2_sr, dtype=float)

    if normalization == "none":
        return dcs

    if normalization == "geom":
        G = geometric_cross_section_m2(scattering_result)
        return dcs / G

    if normalization == "sca":
        if hasattr(scattering_result, "phase_function_sr_inv"):
            return np.asarray(scattering_result.phase_function_sr_inv, dtype=float)

        csca = _cross_section_array(scattering_result, "sca")
        denom = np.maximum(csca[:, None], 1e-300)
        return dcs / denom

    raise ValueError("normalization must be one of: 'none', 'geom', 'sca'.")


# ============================================================
# Angular plots
# ============================================================

def plot_differential_scattering_vs_angle(
    scattering_result,
    *,
    wavelengths_nm: list[float] | np.ndarray | None = None,
    normalization: str = "geom",
    as_db: bool = False,
    cmap: str = "turbo",
    alpha: float = 0.8,
    linewidth: float = 1.2,
    add_colorbar: bool = True,
    ax=None,
):
    """
    Plot angle-dependent scattering for selected wavelengths.

    Parameters
    ----------
    wavelengths_nm :
        If None, plot all wavelengths.
        Otherwise plot the nearest available wavelength for each value.
    normalization :
        "none", "geom", or "sca"
    as_db :
        If True, convert y-values to dB
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    theta_deg = np.asarray(scattering_result.theta_rad, dtype=float) * 180.0 / np.pi
    wl_nm_all = np.asarray(scattering_result.wavelengths_m, dtype=float) * 1e9
    qang = angular_quantity(scattering_result, normalization=normalization)

    if wavelengths_nm is None:
        indices = np.arange(wl_nm_all.size)
    else:
        indices = [nearest_wavelength_index(scattering_result.wavelengths_m, wl) for wl in wavelengths_nm]

    cmap_obj = cm.get_cmap(cmap)
    norm = colors.Normalize(vmin=float(wl_nm_all[indices].min()), vmax=float(wl_nm_all[indices].max()))

    for idx in indices:
        y = qang[idx]
        if as_db:
            y = to_db(y)

        ax.plot(
            theta_deg,
            y,
            color=cmap_obj(norm(wl_nm_all[idx])),
            alpha=alpha,
            linewidth=linewidth,
        )

    ax.set_xlabel("Scattering angle θ [deg]")

    if normalization == "none":
        ylabel = r"$d\sigma/d\Omega$ [m$^2$ sr$^{-1}$]"
    elif normalization == "geom":
        ylabel = r"$(d\sigma/d\Omega)/(\pi R^2)$ [sr$^{-1}$]"
    else:
        ylabel = r"Phase function [sr$^{-1}$]"

    if as_db:
        ylabel += " [dB]"

    ax.set_ylabel(ylabel)
    ax.set_title("Angle-dependent scattering")

    if add_colorbar:
        sm = ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Wavelength [nm]")

    return ax


def plot_polar_scattering(
    scattering_result,
    *,
    wavelengths_nm: list[float] | np.ndarray | float,
    normalization: str = "geom",
    as_db: bool = False,
    cmap: str = "turbo",
    alpha: float = 0.7,
    linewidth: float = 1.4,
    mirror: bool = False,
    ax=None,
):
    """
    Polar plot of angle-dependent scattering for one or more wavelengths.

    Parameters
    ----------
    wavelengths_nm :
        A single wavelength or list of wavelengths (nm)
    normalization :
        "none", "geom", or "sca"
    as_db :
        If True, convert the radial quantity to dB
    mirror :
        If True, mirror the pattern from [0, π] to [0, 2π]
    """
    if np.isscalar(wavelengths_nm):
        wavelengths_nm = [float(wavelengths_nm)]

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))

    theta = np.asarray(scattering_result.theta_rad, dtype=float)
    wl_nm_all = np.asarray(scattering_result.wavelengths_m, dtype=float) * 1e9
    qang = angular_quantity(scattering_result, normalization=normalization)

    cmap_obj = cm.get_cmap(cmap)

    wmin = float(min(wavelengths_nm))
    wmax = float(max(wavelengths_nm))
    if wmin == wmax:
        wmax = wmin + 1e-9
    norm = colors.Normalize(vmin=wmin, vmax=wmax)

    for wl in wavelengths_nm:
        idx = nearest_wavelength_index(scattering_result.wavelengths_m, wl)
        r = qang[idx]
        if as_db:
            r = to_db(r)

        th = theta
        rr = r

        if mirror:
            th = np.concatenate([theta, 2.0 * np.pi - theta[::-1]])
            rr = np.concatenate([r, r[::-1]])

        ax.plot(
            th,
            rr,
            color=cmap_obj(norm(wl_nm_all[idx])),
            alpha=alpha,
            linewidth=linewidth,
            label=f"{wl_nm_all[idx]:.1f} nm",
        )

    ax.set_title("Polar scattering plot")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=8)

    return ax


# ============================================================
# NA / cone integration
# ============================================================

def na_to_half_angle_rad(na: float, n_medium: float = 1.0) -> float:
    """
    Convert numerical aperture to cone half-angle in radians:
        alpha = arcsin(NA / n_medium)
    """
    if na < 0:
        raise ValueError("NA must be non-negative.")
    if na > n_medium:
        raise ValueError(f"NA={na} exceeds n_medium={n_medium}.")
    return float(np.arcsin(na / n_medium))


def integrate_angle_band(
    scattering_result,
    theta_min_rad: float,
    theta_max_rad: float,
) -> np.ndarray:
    """
    Integrate dσ/dΩ over [theta_min_rad, theta_max_rad].

    Returns
    -------
    c_band_m2 : ndarray, shape (n_wavelengths,)
        Integrated scattering cross-section into the angular band [m²]
    """
    theta = np.asarray(scattering_result.theta_rad, dtype=float)
    dcs = np.asarray(scattering_result.dcs_m2_sr, dtype=float)

    mask = (theta >= theta_min_rad) & (theta <= theta_max_rad)
    theta_sel = theta[mask]
    dcs_sel = dcs[:, mask]

    if theta_sel.size < 2:
        raise ValueError("Angle band contains fewer than 2 theta points. Increase angular resolution.")

    integrand = dcs_sel * (2.0 * np.pi * np.sin(theta_sel))[None, :]
    c_band_m2 = np.trapz(integrand, theta_sel, axis=1)

    return np.asarray(c_band_m2, dtype=float)


def split_forward_side_backward(
    scattering_result,
    *,
    alpha_deg: float | None = None,
    alpha_rad: float | None = None,
    n_medium: float = 1.0,
):
    """
    Split total scattering into:
    - forward cone
    - backward cone
    - side scattering (remainder)

    The cone half-angle is set by either alpha_deg / alpha_rad.

    Returns
    -------
    dict with keys:
        c_forward_m2
        c_backward_m2
        c_side_m2
        q_forward_geom
        q_backward_geom
        q_side_geom
        f_forward_of_sca
        f_backward_of_sca
        f_side_of_sca
    """
    if alpha_rad is None:
        if alpha_deg is None:
            raise ValueError("Provide either alpha_deg or alpha_rad.")
        alpha_rad = np.deg2rad(alpha_deg)

    c_forward = integrate_angle_band(scattering_result, 0.0, alpha_rad)
    c_backward = integrate_angle_band(scattering_result, np.pi - alpha_rad, np.pi)
    c_sca = _cross_section_array(scattering_result, "sca")
    c_side = np.clip(c_sca - c_forward - c_backward, 0.0, None)

    G = geometric_cross_section_m2(scattering_result)

    denom_sca = np.maximum(c_sca, 1e-300)

    return {
        "c_forward_m2": c_forward,
        "c_backward_m2": c_backward,
        "c_side_m2": c_side,
        "q_forward_geom": c_forward / G,
        "q_backward_geom": c_backward / G,
        "q_side_geom": c_side / G,
        "f_forward_of_sca": c_forward / denom_sca,
        "f_backward_of_sca": c_backward / denom_sca,
        "f_side_of_sca": c_side / denom_sca,
    }


# ============================================================
# Wavelength plots of forward / side / backward
# ============================================================

def plot_forward_side_backward_vs_wavelength(
    scattering_result,
    *,
    alpha_deg: float,
    normalize: str = "none",
    as_db: bool = False,
    ax=None,
):
    """
    Plot forward / side / backward integrated spectra.

    Parameters
    ----------
    normalize :
        "none" : integrated cross-sections [m²]
        "geom" : divide by geometric cross-section πR²
        "sca"  : divide by total scattering cross-section (fractions of scattering)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))

    wl_nm = np.asarray(scattering_result.wavelengths_m, dtype=float) * 1e9
    bands = split_forward_side_backward(scattering_result, alpha_deg=alpha_deg)

    if normalize == "none":
        y_f = bands["c_forward_m2"]
        y_b = bands["c_backward_m2"]
        y_s = bands["c_side_m2"]
        ylabel = "Integrated cross-section [m²]"
    elif normalize == "geom":
        y_f = bands["q_forward_geom"]
        y_b = bands["q_backward_geom"]
        y_s = bands["q_side_geom"]
        ylabel = "Integrated efficiency [-]"
    elif normalize == "sca":
        y_f = bands["f_forward_of_sca"]
        y_b = bands["f_backward_of_sca"]
        y_s = bands["f_side_of_sca"]
        ylabel = "Fraction of total scattering [-]"
    else:
        raise ValueError("normalize must be one of: 'none', 'geom', 'sca'.")

    if as_db:
        y_f = to_db(y_f)
        y_b = to_db(y_b)
        y_s = to_db(y_s)
        ylabel += " [dB]"

    ax.plot(wl_nm, y_f, label="Forward", linewidth=1.8)
    ax.plot(wl_nm, y_s, label="Side", linewidth=1.8)
    ax.plot(wl_nm, y_b, label="Backward", linewidth=1.8)

    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Forward / side / backward decomposition (α = {alpha_deg:.1f}°)")
    ax.legend()

    return ax


# ============================================================
# Pie chart at selected wavelength
# ============================================================

def plot_partition_pie_at_wavelength(
    scattering_result,
    *,
    alpha_deg: float,
    wavelength_nm: float | None = None,
    design_wavelength_nm: float | None = None,
    include_absorption: bool = True,
    ax=None,
):
    """
    Plot a pie chart of forward / side / backward / absorption fractions
    at a selected wavelength.

    Fractions are normalized to total extinction:
        cext = csca + cabs

    If wavelength_nm is None and design_wavelength_nm is provided, the design wavelength is used.

    Small negative numerical noise is clipped to zero.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))

    wl_nm_all = np.asarray(scattering_result.wavelengths_m, dtype=float) * 1e9

    if wavelength_nm is None:
        if design_wavelength_nm is not None:
            wavelength_nm = float(design_wavelength_nm)
        else:
            # fallback: wavelength of maximum total scattering
            c_sca = _cross_section_array(scattering_result, "sca")
            wavelength_nm = float(wl_nm_all[np.argmax(c_sca)])
            # wavelength_nm = float(wl_nm_all[np.argmax(scattering_result.qsca)])

    idx = nearest_wavelength_index(scattering_result.wavelengths_m, wavelength_nm)
    wl_sel = float(wl_nm_all[idx])

    bands = split_forward_side_backward(scattering_result, alpha_deg=alpha_deg)

    c_forward = max(float(bands["c_forward_m2"][idx]), 0.0)
    c_backward = max(float(bands["c_backward_m2"][idx]), 0.0)
    c_side = max(float(bands["c_side_m2"][idx]), 0.0)
    c_abs = max(float(_cross_section_array(scattering_result, "abs")[idx]), 0.0)
    c_ext = max(float(_cross_section_array(scattering_result, "ext")[idx]), 0.0)

    values = [c_forward, c_side, c_backward]
    labels = ["Forward", "Side", "Backward"]

    if include_absorption:
        values.append(c_abs)
        labels.append("Absorption")

    total = sum(values)
    if total <= 0:
        raise ValueError("All components are zero after clipping.")

    fractions = np.asarray(values, dtype=float) / total

    ax.pie(
        fractions,
        labels=[f"{lab}\n{100*frac:.1f}%" for lab, frac in zip(labels, fractions)],
        autopct=None,
        startangle=90,
        counterclock=False,
    )
    ax.set_title(f"Partition at {wl_sel:.1f} nm (α = {alpha_deg:.1f}°)")

    return ax


# ============================================================
# Metrics reporting
# ============================================================

def scattering_metrics_dataframe(
    scattering_result,
    *,
    wavelength_nm: float | None = None,
    design_wavelength_nm: float | None = None,
    alpha_deg: float | None = None,
) -> pd.DataFrame:
    """
    Return a one-row DataFrame of useful scattering metrics at a selected wavelength.

    Parameters
    ----------
    wavelength_nm :
        Explicit wavelength in nm
    design_wavelength_nm :
        Used if wavelength_nm is None
    alpha_deg :
        If given, also reports forward/side/backward scattering fractions
        for that cone half-angle
    """
    wl_nm_all = np.asarray(scattering_result.wavelengths_m, dtype=float) * 1e9

    if wavelength_nm is None:
        if design_wavelength_nm is not None:
            wavelength_nm = float(design_wavelength_nm)
        else:
            c_sca = _cross_section_array(scattering_result, "sca")
            wavelength_nm = float(wl_nm_all[np.argmax(c_sca)])

    idx = nearest_wavelength_index(scattering_result.wavelengths_m, wavelength_nm)
    wl_sel = float(wl_nm_all[idx])

    G = geometric_cross_section_m2(scattering_result)

    cext = _cross_section_array(scattering_result, "ext")[idx]
    csca = _cross_section_array(scattering_result, "sca")[idx]
    cabs = _cross_section_array(scattering_result, "abs")[idx]

    qext = float(np.asarray(scattering_result.qext, dtype=float)[idx])
    qsca = float(np.asarray(scattering_result.qsca, dtype=float)[idx])
    qabs = float(np.asarray(scattering_result.qabs, dtype=float)[idx])
    qbk = float(np.asarray(scattering_result.qbk, dtype=float)[idx])

    g = float(np.asarray(scattering_result.g, dtype=float)[idx])
    albedo = float(np.asarray(scattering_result.albedo, dtype=float)[idx])

    if g > 0.3:
        g_regime = "forward-biased"
    elif g < -0.3:
        g_regime = "backward-biased"
    else:
        g_regime = "weakly anisotropic / near-symmetric"

    if albedo > 0.9:
        albedo_regime = "mostly scattering"
    elif albedo < 0.1:
        albedo_regime = "mostly absorbing"
    else:
        albedo_regime = "mixed scattering/absorption"

    row = {
        "wavelength_nm": wl_sel,
        "outer_radius_nm": float(np.asarray(scattering_result.radii_m, dtype=float)[-1] * 1e9),
        "geometric_cross_section_m2": G,
        "cext_m2": cext,
        "csca_m2": csca,
        "cabs_m2": cabs,
        "qext": qext,
        "qsca": qsca,
        "qabs": qabs,
        "qbk": qbk,
        "g": g,
        "g_regime": g_regime,
        "albedo": albedo,
        "albedo_regime": albedo_regime,
    }

    if alpha_deg is not None:
        bands = split_forward_side_backward(scattering_result, alpha_deg=alpha_deg)
        row["forward_fraction_of_sca"] = float(bands["f_forward_of_sca"][idx])
        row["side_fraction_of_sca"] = float(bands["f_side_of_sca"][idx])
        row["backward_fraction_of_sca"] = float(bands["f_backward_of_sca"][idx])

    return pd.DataFrame([row])


def print_scattering_metrics(
    scattering_result,
    *,
    wavelength_nm: float | None = None,
    design_wavelength_nm: float | None = None,
    alpha_deg: float | None = None,
) -> None:
    """
    Print a compact metrics summary for one wavelength.
    """
    df = scattering_metrics_dataframe(
        scattering_result,
        wavelength_nm=wavelength_nm,
        design_wavelength_nm=design_wavelength_nm,
        alpha_deg=alpha_deg,
    )
    print(df.to_string(index=False))