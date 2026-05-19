# scattering_sectors.py

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CORE INTEGRATION (already exists)
# ============================================================
def integrate_sectors(theta_rad, dcs_geom, alpha_deg):
    alpha = np.deg2rad(alpha_deg)
    sin_t = np.sin(theta_rad)

    fwd = theta_rad <= alpha
    bwd = theta_rad >= (np.pi - alpha)
    side = (~fwd) & (~bwd)

    def integrate(mask):
        return np.array([
            np.trapezoid(curve[mask] * sin_t[mask], theta_rad[mask])
            for curve in dcs_geom
        ])

    return {
        "forward": integrate(fwd),
        "side": integrate(side),
        "backward": integrate(bwd),
    }


# ============================================================
# PLOT: spectral sector curves
# ============================================================
def plot_scattering_sectors(
    wavelengths_nm,
    sector_dict,
    *,
    alpha_deg,
    y_mode="db",
):
    fwd = sector_dict["forward"]
    side = sector_dict["side"]
    bwd = sector_dict["backward"]

    stack = np.vstack([fwd, side, bwd])

    if y_mode == "db":
        vals = 10 * np.log10(np.maximum(stack, 1e-30))
        vals -= np.nanmax(vals)
        ylabel = "Sector-integrated scattering (dB, normalized)"
    elif y_mode == "linear":
        vals = stack
        ylabel = "Sector-integrated scattering"
    else:
        raise ValueError("y_mode must be 'db' or 'linear'")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(wavelengths_nm, vals[0], label=f"Forward (0–{alpha_deg:.1f}°)")
    ax.plot(wavelengths_nm, vals[1], label="Side")
    ax.plot(wavelengths_nm, vals[2], label=f"Backward ({180-alpha_deg:.1f}–180°)")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Forward / side / backward scattering (α = {alpha_deg:.1f}°)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


# ============================================================
# PLOT: pie chart with optional absorption
# ============================================================
def plot_scattering_pie(
    wavelengths_nm,
    sector_dict,
    *,
    wavelength_nm,
    alpha_deg,
    include_absorption=False,
    qsca=None,
    qabs=None,
):
    # -------- select wavelength index ----------
    iw = int(np.argmin(np.abs(wavelengths_nm - wavelength_nm)))

    fwd = sector_dict["forward"][iw]
    side = sector_dict["side"][iw]
    bwd = sector_dict["backward"][iw]

    vals = [fwd, side, bwd]
    labels = [
        f"Forward\n(0–{alpha_deg:.1f}°)",
        "Side",
        f"Backward\n({180-alpha_deg:.1f}–180°)",
    ]

    # -------- optional absorption ----------
    if include_absorption:
        if qsca is None or qabs is None:
            raise ValueError(
                "qsca and qabs must be provided when include_absorption=True"
            )

        qsca_val = qsca[iw]
        qabs_val = qabs[iw]

        # absolute contribution (physical, not normalized yet)
        absorption = qabs_val

        vals.append(absorption)
        labels.append("Absorption")

    # -------- normalize ----------
    vals = np.array(vals, dtype=float)

    if np.sum(vals) > 0:
        vals = vals / np.sum(vals)

    # -------- plot ----------
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.pie(
        vals,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
    )

    ax.set_title(
        f"Angular fractions at {wavelengths_nm[iw]:.1f} nm\n"
        f"(α = {alpha_deg:.1f}°)"
    )

    return fig, ax

import numpy as np
import matplotlib.pyplot as plt


def plot_backward_alpha_sweep(
    theta_rad,
    dcs_geom,
    wavelengths_nm,
    *,
    alpha_list=(10, 30, 50, 70.5, 90.0),
    row_case=None,
    normalize=True,
):
    """
    Plot backward-integrated scattering vs wavelength for different α.

    Shows:
      - absolute curves (left axis)
      - normalized curves (right axis)

    Parameters
    ----------
    theta_rad : array
    dcs_geom : array (n_wavelengths, n_theta)
    wavelengths_nm : array
    alpha_list : iterable
    row_case : pandas.Series, optional
    normalize : bool
        plot normalized versions on second axis
    """

    theta = np.asarray(theta_rad, dtype=float)
    sin_theta = np.sin(theta)

    # ------------------------------------------------------------
    # Compute backward integrals
    # ------------------------------------------------------------
    def backward_integral(alpha_deg):
        alpha_rad = np.deg2rad(alpha_deg)
        mask = theta >= (np.pi - alpha_rad)

        vals = []
        for curve in dcs_geom:
            th = theta[mask]
            yy = np.asarray(curve, dtype=float)[mask] * sin_theta[mask]
            vals.append(np.trapezoid(yy, th))

        return np.asarray(vals, dtype=float)

    sector_curves = {a: backward_integral(a) for a in alpha_list}

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 5))

    if normalize:
        ax2 = ax1.twinx()
    else:
        ax2 = None

    for a in alpha_list:
        yy = sector_curves[a]

        # absolute
        ax1.plot(
            wavelengths_nm,
            yy,
            linewidth=2.2,
            linestyle="--",
            label=f"α = {a:.1f}°",
        )

        # normalized
        if normalize:
            ymax = np.nanmax(yy)
            yy_norm = yy / ymax if ymax > 0 else yy

            ax2.plot(
                wavelengths_nm,
                yy_norm,
                linewidth=2.2,
                label=f"α = {a:.1f}° (norm.)",
            )

    # ------------------------------------------------------------
    # Reference line
    # ------------------------------------------------------------
    if row_case is not None and "design_peak_wavelength_nm" in row_case.index:
        ax1.axvline(
            float(row_case["design_peak_wavelength_nm"]),
            color="gray",
            linestyle="--",
            linewidth=1.0,
            label="design λ",
        )

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_xlim(wavelengths_nm[0], wavelengths_nm[-1])
    ax1.set_ylabel("Backward-integrated scattering")

    if normalize:
        ax2.set_ylabel("Normalized backward scattering")
        ax2.set_ylim(0, None)

    ax1.set_ylim(0, None)

    ax1.set_title("Effect of backward collection half-angle α on spectral peak position")

    ax1.grid(True, alpha=0.3)

    # ------------------------------------------------------------
    # Legend handling
    # ------------------------------------------------------------
    if normalize:
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend(loc="best")

    fig.tight_layout()
    return fig, (ax1, ax2)