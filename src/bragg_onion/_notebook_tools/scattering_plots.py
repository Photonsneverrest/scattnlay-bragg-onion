# scattering_plots.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_differential_vs_theta(
    theta_deg,
    wavelengths_nm,
    dcs_geom,
    *,
    plot_all=True,
    y_mode="db",
    normalize_each_curve=True,
    # styling
    all_alpha=0.18,
    all_cmap="turbo",
    all_linewidth=1.0,
    selected_linewidth=2.0,
    selected_step=50,
    show_legend=False,
):
    """
    Plot differential scattering vs theta.

    Parameters
    ----------
    theta_deg : array
    wavelengths_nm : array
    dcs_geom : array (n_wavelengths, n_theta)
    plot_all : bool
        If True: plot all wavelengths with colormap
        If False: plot subsampled set
    y_mode : "db" or "linear"
    normalize_each_curve : bool
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)

    # ------------------------------------------------------------
    # Prepare colormap for all-wavelength mode
    # ------------------------------------------------------------
    cmap = mpl.colormaps[all_cmap]
    norm = mpl.colors.Normalize(
        vmin=float(np.min(wavelengths_nm)),
        vmax=float(np.max(wavelengths_nm)),
    )

    # ------------------------------------------------------------
    # Loop over wavelengths
    # ------------------------------------------------------------
    for iw in range(len(wavelengths_nm)):
        curve = np.asarray(dcs_geom[iw], dtype=float)

        # Normalize
        if normalize_each_curve:
            cmax = np.nanmax(curve)
            if cmax > 0:
                curve = curve / cmax

        # Convert to chosen y-scale
        if y_mode == "db":
            curve = 10.0 * np.log10(np.maximum(curve, 1e-30))
            if not normalize_each_curve:
                curve -= np.nanmax(curve)
        elif y_mode == "linear":
            pass
        else:
            raise ValueError("y_mode must be 'db' or 'linear'.")

        # --------------------------------------------------------
        # Plotting
        # --------------------------------------------------------
        if plot_all:
            ax.plot(
                theta_deg,
                curve,
                color=cmap(norm(wavelengths_nm[iw])),
                alpha=all_alpha,
                linewidth=all_linewidth,
            )
        else:
            if iw % selected_step == 0:
                ax.plot(
                    theta_deg,
                    curve,
                    linewidth=selected_linewidth,
                    label=f"{wavelengths_nm[iw]:.1f} nm",
                )

    # ------------------------------------------------------------
    # Colorbar (only in all-wavelength mode)
    # ------------------------------------------------------------
    if plot_all:
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Wavelength (nm)")

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------
    ax.set_xlabel("θ (deg)")

    if y_mode == "db":
        ylabel = "Differential scattering (dB"
    else:
        ylabel = "Differential scattering (linear"

    if normalize_each_curve:
        ylabel += ", normalized)"
    else:
        ylabel += ")"

    ax.set_ylabel(ylabel)
    ax.set_title("Differential scattering vs θ")

    ax.set_xlim(0, 180)
    ax.grid(True, alpha=0.3)

    if not plot_all and show_legend:
        ax.legend()

    fig.tight_layout()

    return fig, ax


def plot_polar_differential_scattering(
    theta_deg,
    wavelengths_nm,
    dcs_geom,
    *,
    # modes
    plot_all=True,
    y_mode="db",
    normalize_each_curve=True,
    # wavelength selection (for plot_all=False)
    selected_wavelengths_nm=None,
    selected_step=50,
    show_legend=False,
    # styling
    all_cmap="turbo",
    all_alpha=0.12,
    all_linewidth=0.9,
    selected_linewidth=2.0,
    # axis options
    show_signed_ticks=True,
):
    """
    Polar differential scattering plot (mirrored 0..180° to ±180°).

    Parameters
    ----------
    theta_deg : array
    wavelengths_nm : array
    dcs_geom : array (n_wavelengths, n_theta)
    """

    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    theta_deg = np.asarray(theta_deg, dtype=float)
    theta_rad = np.deg2rad(theta_deg)

    # ------------------------------------------------------------
    # Mirror theta to negative side
    # ------------------------------------------------------------
    theta_neg_rad = -theta_rad[-2:0:-1]
    theta_full_rad = np.concatenate([theta_neg_rad, theta_rad])

    # ------------------------------------------------------------
    # Select wavelengths
    # ------------------------------------------------------------
    if plot_all:
        wavelength_indices = list(range(len(wavelengths_nm)))
        title_suffix = "all wavelengths"
    else:
        if selected_wavelengths_nm is None:
            wavelength_indices = list(range(0, len(wavelengths_nm), selected_step))
        else:
            def nearest_idx(val):
                return int(np.argmin(np.abs(wavelengths_nm - val)))

            wavelength_indices = [nearest_idx(w) for w in selected_wavelengths_nm]

        title_suffix = "selected wavelengths"

    # ------------------------------------------------------------
    # Build curves
    # ------------------------------------------------------------
    curves = []
    labels = []

    for iw in wavelength_indices:
        curve = np.asarray(dcs_geom[iw], dtype=float)

        # normalize
        if normalize_each_curve:
            cmax = np.nanmax(curve)
            if cmax > 0:
                curve = curve / cmax

        # y scaling
        if y_mode == "db":
            yvals = 10.0 * np.log10(np.maximum(curve, 1e-30))
            if not normalize_each_curve:
                yvals -= np.nanmax(yvals)
        elif y_mode == "linear":
            yvals = curve
        else:
            raise ValueError("y_mode must be 'db' or 'linear'.")

        # mirror across x-axis
        y_mirror = yvals[-2:0:-1]
        y_full = np.concatenate([y_mirror, yvals])

        curves.append(y_full)
        labels.append(f"{wavelengths_nm[iw]:.1f} nm")

    curves = np.asarray(curves)

    # ------------------------------------------------------------
    # Determine radial scaling
    # ------------------------------------------------------------
    finite_vals = curves[np.isfinite(curves)]

    if finite_vals.size == 0:
        y_min, y_max = (-40, 0) if y_mode == "db" else (0, 1)
    else:
        if y_mode == "db":
            y_min = np.floor(np.nanpercentile(finite_vals, 1)/5)*5
            y_min = max(y_min, -60)
            y_max = 0 if normalize_each_curve else np.nanmax(finite_vals)
        else:
            y_min = 0
            y_max = np.nanmax(finite_vals)

    # ------------------------------------------------------------
    # Create plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    # ticks
    if show_signed_ticks:
        tick_deg = [0, 45, 90, 135, 180, 225, 270, 315]
        tick_lbl = ["0°", "45°", "90°", "135°", "180°", "-135°", "-90°", "-45°"]
    else:
        tick_deg = [0, 90, 180, 270]
        tick_lbl = ["0°", "90°", "180°", "-90°"]

    ax.set_xticks(np.deg2rad(tick_deg))
    ax.set_xticklabels(tick_lbl)

    # ------------------------------------------------------------
    # Plot curves
    # ------------------------------------------------------------
    if plot_all:
        cmap = mpl.colormaps[all_cmap]
        norm = mpl.colors.Normalize(vmin=np.min(wavelengths_nm), vmax=np.max(wavelengths_nm))

        for i, iw in enumerate(wavelength_indices):
            yvals = curves[i]
            r = yvals - y_min if y_mode == "db" else yvals

            ax.plot(
                theta_full_rad,
                r,
                color=cmap(norm(wavelengths_nm[iw])),
                alpha=all_alpha,
                linewidth=all_linewidth,
            )

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.1)
        cbar.set_label("Wavelength (nm)")

    else:
        for i, iw in enumerate(wavelength_indices):
            yvals = curves[i]
            r = yvals - y_min if y_mode == "db" else yvals

            ax.plot(
                theta_full_rad,
                r,
                linewidth=selected_linewidth,
                label=labels[i],
            )

        if show_legend:
            ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))

    # ------------------------------------------------------------
    # Radial axis
    # ------------------------------------------------------------
    if y_mode == "db":
        rmax = y_max - y_min
        ax.set_ylim(0, rmax)

        ticks_db = np.linspace(y_min, y_max, 5)
        ax.set_yticks(ticks_db - y_min)
        ax.set_yticklabels([f"{t:.0f} dB" for t in ticks_db])
    else:
        ax.set_ylim(y_min, y_max)

    # ------------------------------------------------------------
    # Title
    # ------------------------------------------------------------
    norm_txt = "normalized" if normalize_each_curve else "absolute"

    ax.set_title(
        f"Polar differential scattering ({y_mode}, {norm_txt})\n{title_suffix}",
        va="bottom",
    )

    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax

def plot_integrated_efficiencies(
    wavelengths_nm,
    *,
    qsca=None,
    qext=None,
    qabs=None,
    wavelengths_int_nm=None,
    q_collected_geom=None,
    row_case=None,
    title="Integrated efficiencies vs wavelength",
):
    """
    Plot integrated scattering efficiencies vs wavelength.

    Parameters
    ----------
    wavelengths_nm : array
        wavelength grid for qsca/qext/qabs
    wavelengths_int_nm : array, optional
        wavelength grid for q_collected_geom
    qsca, qext, qabs : arrays, optional
    q_collected_geom : array, optional
    row_case : pandas.Series, optional
        used to annotate design / spectrum peak wavelengths
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    # ------------------------------------------------------------
    # Plot efficiencies
    # ------------------------------------------------------------
    if qsca is not None:
        ax.plot(wavelengths_nm, qsca, label="Qsca")

    if qext is not None:
        ax.plot(wavelengths_nm, qext, label="Qext")

    if qabs is not None:
        ax.plot(wavelengths_nm, qabs, label="Qabs")

    if q_collected_geom is not None and wavelengths_int_nm is not None:
        ax.plot(wavelengths_int_nm, q_collected_geom, label="Qcollected,geom")

    # ------------------------------------------------------------
    # Mark reference wavelengths
    # ------------------------------------------------------------
    if row_case is not None:
        # design wavelength
        if "design_peak_wavelength_nm" in row_case.index:
            ax.axvline(
                float(row_case["design_peak_wavelength_nm"]),
                color="gray",
                linestyle="--",
                linewidth=1.0,
                label="design λ",
            )

        # spectrum peak
        if "spectrum_peak_wavelength_nm" in row_case.index:
            ax.axvline(
                float(row_case["spectrum_peak_wavelength_nm"]),
                color="tab:red",
                linestyle=":",
                linewidth=1.0,
                label="spectrum peak λ",
            )

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Efficiency")
    ax.set_title(title)

    ax.grid(True, alpha=0.3)

    # only show legend if something is plotted
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()

    fig.tight_layout()
    return fig, ax

def plot_integrated_with_g_albedo(
    data,
    row_case=None,
    *,
    integ=None,
    show_diagnostics=False,
):
    """
    Plot Qcollected, g, and albedo vs wavelength.

    Parameters
    ----------
    data : dict
        output from extract_case_arrays(...)
    row_case : pandas.Series, optional
        used to mark design / spectrum wavelengths
    integ : integrated_result object, optional
        used for cone information in title
    show_diagnostics : bool
        print shape / integration info
    """

    # ------------------------------------------------------------
    # Data extraction + flattening
    # ------------------------------------------------------------
    wl_nm = np.asarray(data["wavelengths_int_nm"], dtype=float).ravel()
    qcol = np.asarray(data["q_collected_geom"], dtype=float).ravel()

    g = np.asarray(data["g"], dtype=float).ravel()
    alb = np.asarray(data["albedo"], dtype=float).ravel()

    wl_nm_scat = np.asarray(data["wavelengths_nm"], dtype=float).ravel()

    # ------------------------------------------------------------
    # Optional diagnostics
    # ------------------------------------------------------------
    if show_diagnostics:
        print("Shapes:")
        print("  wl_nm:", wl_nm.shape)
        print("  qcol:", qcol.shape)
        print("  wl_nm_scat:", wl_nm_scat.shape)
        print("  g:", g.shape)
        print("  albedo:", alb.shape)

    # ------------------------------------------------------------
    # Integration cone info (if available)
    # ------------------------------------------------------------
    title_extra = ""

    if integ is not None:
        theta_min_deg_arr = np.rad2deg(np.asarray(integ.theta_min_rad)).ravel()
        theta_max_deg_arr = np.rad2deg(np.asarray(integ.theta_max_rad)).ravel()

        theta_min_deg = float(theta_min_deg_arr[0])
        theta_max_deg = float(theta_max_deg_arr[0])

        alpha_deg = np.rad2deg(np.arcsin(float(integ.collection_na)))

        title_extra = (
            f"\n(direction = {integ.direction}, "
            f"α = {alpha_deg:.1f}°, "
            f"θ-range = {theta_min_deg:.1f}°–{theta_max_deg:.1f}°)"
        )

        if show_diagnostics:
            print("\nIntegration:")
            print("  direction:", integ.direction)
            print("  alpha:", alpha_deg)
            print("  theta range:", theta_min_deg, theta_max_deg)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Left axis (Qcollected)
    ax1.plot(
        wl_nm,
        qcol,
        color="black",
        linewidth=2.2,
        label="Qcollected,geom",
    )

    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_xlim(wl_nm[0], wl_nm[-1])
    ax1.set_ylabel("Integrated scattering efficiency", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(True, alpha=0.3)

    # Right axis (g + albedo)
    ax2 = ax1.twinx()

    ax2.plot(
        wl_nm_scat,
        g,
        color="tab:blue",
        linewidth=1.8,
        label="g",
    )

    ax2.plot(
        wl_nm_scat,
        alb,
        color="tab:orange",
        linewidth=1.8,
        label="albedo",
    )

    ax2.set_ylabel("g / albedo")

    # ------------------------------------------------------------
    # Reference lines
    # ------------------------------------------------------------
    if row_case is not None:
        if "design_peak_wavelength_nm" in row_case.index:
            ax1.axvline(
                float(row_case["design_peak_wavelength_nm"]),
                color="gray",
                linestyle="--",
                linewidth=1.0,
                label="design λ",
            )

        if "spectrum_peak_wavelength_nm" in row_case.index:
            ax1.axvline(
                float(row_case["spectrum_peak_wavelength_nm"]),
                color="tab:red",
                linestyle=":",
                linewidth=1.0,
                label="spectrum peak λ",
            )

    # ------------------------------------------------------------
    # Combined legend
    # ------------------------------------------------------------
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="center right",
    )

    # ------------------------------------------------------------
    # Title
    # ------------------------------------------------------------
    ax1.set_title(
        "Integrated collected efficiency, asymmetry g, and albedo vs wavelength"
        + title_extra
    )

    fig.tight_layout()
    return fig, (ax1, ax2)
