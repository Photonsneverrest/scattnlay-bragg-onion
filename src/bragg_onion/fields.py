from __future__ import annotations

# fields.py
#
# Compute and visualize near fields around multilayer Bragg onion spheres.
#
# Key features:
# - wraps scattnlay.fieldnlay
# - builds 2D field maps in xy / xz / yz planes
# - computes |E|, |H|
# - computes time-averaged Poynting vectors:
#       S = 0.5 * Re(E x conj(H))
# - reconstructs a simple incident plane wave
# - provides total, scattered-like, and delta-flow visualizations
# - plots field magnitude maps
# - plots Poynting streamlines
# - overlays layer boundaries
# - supports optional streamline masking inside the sphere
# - supports optional custom streamline seed points
#
# Conventions:
# - all geometry is specified in meters
# - scattnlay uses dimensionless coordinates consistent with x = k * r
# - this module converts physical coordinates [m] to dimensionless coordinates
#   using the same real-valued medium wavevector used for x
#
# Notes:
# - The "scattered-like" field is defined here as:
#       E_sca_like = E_total - E_inc
#       H_sca_like = H_total - H_inc
#   using a reconstructed incident plane wave.
# - This is very useful for visualization, but the full total Poynting
#   decomposition also contains interference terms.
# - The "delta" flow is:
#       S_delta = S_total - S_inc
#   which is often the most intuitive visualization of how the sphere modifies
#   the incident power flow.
# - If the surrounding medium has a non-negligible imaginary part, this module
#   still uses Re(k_medium) for coordinate scaling and incident-wave reconstruction.

from dataclasses import dataclass
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from numpy.typing import NDArray

try:
    import scattnlay  # type: ignore
except ImportError as exc:
    raise ImportError(
        "scattnlay is required for bragg_onion.fields. "
        "Install the optional dependency with: pip install 'bragg-onion[scattnlay]'"
    ) from exc

from .geometry import ResolvedLayerStack
from .solver import build_scattnlay_inputs_single_wavelength


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

Plane = Literal["xy", "xz", "yz"]
FieldQuantity = Literal["E", "H", "S"]
FieldKind = Literal["total", "scattered"]
FlowKind = Literal["total", "scattered", "delta"]
ScaleMode = Literal["linear", "db"]

# Vacuum impedance
# from scipy import constants
# Z0_OHM = constants.physical_constants["characteristic impedance of vacuum"]
Z0_OHM = 376.730313668

__all__ = [
    "FieldMapResult",
    "make_line_seeds",
    "compute_field_map",
    "plot_field_magnitude",
    "plot_poynting_streamlines",
    "plot_poynting_vectors",
]


# ============================================================
# Resolve fieldnlay dynamically
# ============================================================

_fieldnlay = getattr(scattnlay, "fieldnlay", None)
if _fieldnlay is None:
    raise ImportError(
        "The installed scattnlay module does not expose 'fieldnlay'. "
        "Check your scattnlay installation and Python environment."
    )


# ============================================================
# Dataclass
# ============================================================

@dataclass(frozen=True)
class FieldMapResult:
    """
    Container for a computed 2D field map.

    Attributes
    ----------
    wavelength_m :
        Wavelength [m]
    plane :
        Plane used for the map: "xy", "xz", or "yz"
    radii_m :
        Cumulative layer radii [m]
    coord1_m, coord2_m :
        1D coordinate axes in the chosen plane [m]
    grid1_m, grid2_m :
        2D coordinate meshes [m]
    radial_distance_m :
        Radial distance in the selected plane [m]

    e_total, h_total :
        Total complex fields, shape (npts, npts, 3)
    e_inc, h_inc :
        Reconstructed incident plane-wave fields, shape (npts, npts, 3)
    e_sca_like, h_sca_like :
        Scattered-like fields = total - incident, shape (npts, npts, 3)

    e_total_mag, h_total_mag :
        Total field magnitudes
    e_sca_like_mag, h_sca_like_mag :
        Scattered-like field magnitudes

    s_total, s_inc, s_sca_like, s_delta :
        Time-averaged Poynting vectors, shape (npts, npts, 3)
    s_total_mag, s_inc_mag, s_sca_like_mag, s_delta_mag :
        Corresponding magnitudes

    plane_u_total, plane_v_total :
        In-plane components of total Poynting vector
    plane_u_sca_like, plane_v_sca_like :
        In-plane components of scattered-like Poynting vector
    plane_u_delta, plane_v_delta :
        In-plane components of delta-flow vector

    extent_m :
        Plot extent half-width [m]
    """
    wavelength_m: float
    plane: Plane
    radii_m: FloatArray

    coord1_m: FloatArray
    coord2_m: FloatArray
    grid1_m: FloatArray
    grid2_m: FloatArray
    radial_distance_m: FloatArray

    e_total: ComplexArray
    h_total: ComplexArray
    e_inc: ComplexArray
    h_inc: ComplexArray
    e_sca_like: ComplexArray
    h_sca_like: ComplexArray

    e_total_mag: FloatArray
    h_total_mag: FloatArray
    e_sca_like_mag: FloatArray
    h_sca_like_mag: FloatArray

    s_total: FloatArray
    s_inc: FloatArray
    s_sca_like: FloatArray
    s_delta: FloatArray

    s_total_mag: FloatArray
    s_inc_mag: FloatArray
    s_sca_like_mag: FloatArray
    s_delta_mag: FloatArray

    plane_u_total: FloatArray
    plane_v_total: FloatArray
    plane_u_sca_like: FloatArray
    plane_v_sca_like: FloatArray
    plane_u_delta: FloatArray
    plane_v_delta: FloatArray

    extent_m: float

    @property
    def outer_radius_m(self) -> float:
        return float(self.radii_m[-1])

    @property
    def extent_nm(self) -> float:
        return float(self.extent_m * 1e9)


# ============================================================
# Helpers
# ============================================================

def _apply_scale(values: np.ndarray, scale: ScaleMode, floor: float = 1e-30) -> np.ndarray:
    """
    Apply linear or dB scaling.
    """
    arr = np.asarray(values, dtype=float)
    if scale == "linear":
        return arr
    if scale == "db":
        return 10.0 * np.log10(np.maximum(arr, floor))
    raise ValueError(f"Unknown scale: {scale!r}")


def _build_plane_grid(
    plane: Plane,
    extent_m: float,
    npts: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    Build 2D coordinate grids in a chosen plane.

    Returns
    -------
    coord1_m, coord2_m :
        1D axes
    grid1_m, grid2_m :
        2D mesh grids
    x_m, y_m, z_m :
        Flattened physical coordinates [m]
    """
    coord1_m = np.linspace(-extent_m, extent_m, npts, dtype=float)
    coord2_m = np.linspace(-extent_m, extent_m, npts, dtype=float)
    grid1_m, grid2_m = np.meshgrid(coord1_m, coord2_m, indexing="xy")

    if plane == "xy":
        x_m = grid1_m.ravel()
        y_m = grid2_m.ravel()
        z_m = np.zeros_like(x_m)
    elif plane == "xz":
        x_m = grid1_m.ravel()
        z_m = grid2_m.ravel()
        y_m = np.zeros_like(x_m)
    elif plane == "yz":
        y_m = grid1_m.ravel()
        z_m = grid2_m.ravel()
        x_m = np.zeros_like(y_m)
    else:
        raise ValueError(f"Unsupported plane: {plane!r}")

    return coord1_m, coord2_m, grid1_m, grid2_m, x_m, y_m, z_m


def _reshape_vector_field(values: np.ndarray, npts: int) -> ComplexArray:
    """
    Reshape flattened vector field to (npts, npts, 3).
    """
    arr = np.asarray(values, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"Expected flattened vector field of shape (N, 3), got {arr.shape}."
        )
    return arr.reshape((npts, npts, 3))


def _vector_magnitude(field: ComplexArray) -> FloatArray:
    """
    Compute vector magnitude sqrt(sum |component|^2).
    """
    return np.sqrt(np.sum(np.abs(field) ** 2, axis=-1)).astype(np.float64)


def _compute_poynting_vector(e_field: ComplexArray, h_field: ComplexArray) -> FloatArray:
    """
    Compute time-averaged Poynting vector:
        S = 0.5 * Re(E x conj(H))
    """
    s = 0.5 * np.real(np.cross(e_field, np.conjugate(h_field)))
    return np.asarray(s, dtype=np.float64)


def _plane_components(
    vector_field: FloatArray,
    plane: Plane,
) -> tuple[FloatArray, FloatArray]:
    """
    Extract in-plane components from a 3D vector field.
    """
    if plane == "xy":
        return vector_field[..., 0], vector_field[..., 1]
    if plane == "xz":
        return vector_field[..., 0], vector_field[..., 2]
    if plane == "yz":
        return vector_field[..., 1], vector_field[..., 2]
    raise ValueError(f"Unsupported plane: {plane!r}")


def _incident_plane_wave(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    *,
    wavelength_m: float,
    n_medium_real: float,
    propagation_axis: Literal["z"] = "z",
    polarization_axis: Literal["x"] = "x",
    e0: complex = 1.0 + 0.0j,
    h0: complex | None = None,
) -> tuple[ComplexArray, ComplexArray]:
    """
    Construct a monochromatic incident plane wave in the surrounding medium.

    Assumptions
    -----------
    - propagation: +z
    - electric polarization: x
    - magnetic field: +y (right-handed plane wave)
    - nonmagnetic medium (μr ≈ 1)

    For a plane wave in a nonmagnetic dielectric medium:
        H0 = E0 / η = (n / Z0) E0
    with η = Z0 / n.

    Returns
    -------
    E_inc, H_inc :
        Arrays of shape (N, 3)
    """
    if propagation_axis != "z":
        raise NotImplementedError("Currently only propagation_axis='z' is implemented.")
    if polarization_axis != "x":
        raise NotImplementedError("Currently only polarization_axis='x' is implemented.")

    k0 = 2.0 * np.pi / wavelength_m
    k_med = k0 * n_medium_real
    phase = np.exp(1j * k_med * z_m)

    if h0 is None:
        h0 = (n_medium_real / Z0_OHM) * e0

    E_inc = np.zeros((x_m.size, 3), dtype=np.complex128)
    H_inc = np.zeros((x_m.size, 3), dtype=np.complex128)

    # +z propagation, E along x, H along +y
    E_inc[:, 0] = e0 * phase
    H_inc[:, 1] = h0 * phase

    return E_inc, H_inc


def _add_layer_boundaries(ax: plt.Axes, radii_m: FloatArray) -> None:
    """
    Overlay layer boundaries as circles in the selected plane.
    """
    for r_m in radii_m:
        circle = Circle(
            (0.0, 0.0),
            radius=float(r_m * 1e9),
            fill=False,
            edgecolor="white",
            linewidth=1.0,
            alpha=0.85,
        )
        ax.add_patch(circle)


def _percentile_limits(
    data: np.ndarray,
    lower: float | None,
    upper: float | None,
) -> tuple[float | None, float | None]:
    """
    Compute percentile-based vmin/vmax if requested.
    """
    arr = np.asarray(data, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return None, None

    vmin = float(np.percentile(arr[finite], lower)) if lower is not None else None
    vmax = float(np.percentile(arr[finite], upper)) if upper is not None else None
    return vmin, vmax

def _reference_speed_for_masking(
    speed: np.ndarray,
    *,
    base_mask: np.ndarray,
    percentile: float = 99.0,
) -> float:
    """
    Compute a robust reference speed from finite, unmasked points
    using a percentile instead of the absolute maximum.
    """
    speed = np.asarray(speed, dtype=float)
    valid = (~base_mask) & np.isfinite(speed)

    if not np.any(valid):
        return 0.0

    return float(np.percentile(speed[valid], percentile))

def make_line_seeds(
    start_nm: tuple[float, float],
    end_nm: tuple[float, float],
    n_seeds: int,
) -> FloatArray:
    """
    Create evenly spaced streamline seed points along a line in the plotting plane.

    Parameters
    ----------
    start_nm, end_nm :
        Start and end point in nm, given in plot coordinates (coord1, coord2)
    n_seeds :
        Number of seeds along the line

    Returns
    -------
    np.ndarray
        Shape (n_seeds, 2), in nm
    """
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1.")

    x0, y0 = start_nm
    x1, y1 = end_nm
    xs = np.linspace(x0, x1, n_seeds, dtype=float)
    ys = np.linspace(y0, y1, n_seeds, dtype=float)
    return np.column_stack([xs, ys]).astype(np.float64)


def _choose_magnitude_array(
    field_result: FieldMapResult,
    *,
    quantity: FieldQuantity,
    kind: Literal["total", "scattered", "delta"],
) -> FloatArray:
    """
    Select the appropriate magnitude array for plotting.
    """
    if quantity == "E":
        if kind == "total":
            return field_result.e_total_mag
        if kind == "scattered":
            return field_result.e_sca_like_mag
        raise ValueError("For quantity='E', kind must be 'total' or 'scattered'.")

    if quantity == "H":
        if kind == "total":
            return field_result.h_total_mag
        if kind == "scattered":
            return field_result.h_sca_like_mag
        raise ValueError("For quantity='H', kind must be 'total' or 'scattered'.")

    if quantity == "S":
        if kind == "total":
            return field_result.s_total_mag
        if kind == "scattered":
            return field_result.s_sca_like_mag
        if kind == "delta":
            return field_result.s_delta_mag

    raise ValueError("Unsupported quantity / kind combination.")


# ============================================================
# Main field computation
# ============================================================

def compute_field_map(
    stack: ResolvedLayerStack,
    *,
    wavelength_m: float,
    n_medium,
    plane: Plane = "xz",
    npts: int = 401,
    extent_m: float | None = None,
    extent_outer_radius_factor: float = 2.2,
    incident_e0: complex = 1.0 + 0.0j,
    incident_h0: complex | None = None,
) -> FieldMapResult:
    """
    Compute a 2D near-field map in a selected plane.

    Parameters
    ----------
    stack :
        Resolved layer stack
    wavelength_m :
        Wavelength [m]
    n_medium :
        Surrounding medium refractive index (scalar or dispersion-like object)
    plane :
        "xy", "xz", or "yz"
    npts :
        Number of grid points per axis
    extent_m :
        Half-width of the plotted region [m]. If None, it is set to
        extent_outer_radius_factor * outer_radius
    extent_outer_radius_factor :
        Used only when extent_m is None
    incident_e0, incident_h0 :
        Complex amplitudes for reconstructed incident E and H fields

    Returns
    -------
    FieldMapResult
    """
    if npts < 5:
        raise ValueError("npts must be at least 5.")
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive.")

    x, m, radii_m, n_medium_eval, k_medium = build_scattnlay_inputs_single_wavelength(
        stack=stack,
        wavelength_m=wavelength_m,
        n_medium=n_medium,
    )

    outer_radius_m = float(radii_m[-1])
    if extent_m is None:
        extent_m = float(extent_outer_radius_factor * outer_radius_m)
    if extent_m <= 0:
        raise ValueError("extent_m must be positive.")

    # Use the same real-valued medium wavevector convention as for x = k * r
    n_medium_real = float(np.real(n_medium_eval))
    if n_medium_real <= 0:
        raise ValueError("Real part of the surrounding-medium refractive index must be positive.")

    if abs(np.imag(n_medium_eval)) > 1e-12:
        print(
            "Warning: n_medium has a non-zero imaginary part. "
            "fields.py currently uses Re(n_medium) for field-grid coordinate scaling "
            "and incident-wave reconstruction."
        )

    k_medium_real = float(np.real(k_medium))

    coord1_m, coord2_m, grid1_m, grid2_m, x_m, y_m, z_m = _build_plane_grid(
        plane=plane,
        extent_m=extent_m,
        npts=npts,
    )

    # Radial distance in the chosen plane
    radial_distance_m = np.sqrt(grid1_m**2 + grid2_m**2)

    # Convert physical coordinates [m] to dimensionless coordinates expected by fieldnlay
    qx = k_medium_real * x_m
    qy = k_medium_real * y_m
    qz = k_medium_real * z_m

    terms, E, H = _fieldnlay(x, m, qx, qy, qz)

    e_total = _reshape_vector_field(E, npts=npts)
    h_total = _reshape_vector_field(H, npts=npts)

    # Reconstruct incident plane wave in physical coordinates
    e_inc_flat, h_inc_flat = _incident_plane_wave(
        x_m, y_m, z_m,
        wavelength_m=wavelength_m,
        n_medium_real=n_medium_real,
        propagation_axis="z",
        polarization_axis="x",
        e0=incident_e0,
        h0=incident_h0,
    )
    e_inc = e_inc_flat.reshape((npts, npts, 3))
    h_inc = h_inc_flat.reshape((npts, npts, 3))

    # Scattered-like field for visualization
    e_sca_like = e_total - e_inc
    h_sca_like = h_total - h_inc

    # Magnitudes
    e_total_mag = _vector_magnitude(e_total)
    h_total_mag = _vector_magnitude(h_total)
    e_sca_like_mag = _vector_magnitude(e_sca_like)
    h_sca_like_mag = _vector_magnitude(h_sca_like)

    # Poynting vectors
    s_total = _compute_poynting_vector(e_total, h_total)
    s_inc = _compute_poynting_vector(e_inc, h_inc)
    s_sca_like = _compute_poynting_vector(e_sca_like, h_sca_like)
    s_delta = s_total - s_inc

    s_total_mag = np.sqrt(np.sum(s_total**2, axis=-1)).astype(np.float64)
    s_inc_mag = np.sqrt(np.sum(s_inc**2, axis=-1)).astype(np.float64)
    s_sca_like_mag = np.sqrt(np.sum(s_sca_like**2, axis=-1)).astype(np.float64)
    s_delta_mag = np.sqrt(np.sum(s_delta**2, axis=-1)).astype(np.float64)

    plane_u_total, plane_v_total = _plane_components(s_total, plane=plane)
    plane_u_sca_like, plane_v_sca_like = _plane_components(s_sca_like, plane=plane)
    plane_u_delta, plane_v_delta = _plane_components(s_delta, plane=plane)

    return FieldMapResult(
        wavelength_m=float(wavelength_m),
        plane=plane,
        radii_m=np.asarray(radii_m, dtype=np.float64),
        coord1_m=coord1_m,
        coord2_m=coord2_m,
        grid1_m=grid1_m,
        grid2_m=grid2_m,
        radial_distance_m=radial_distance_m,
        e_total=e_total,
        h_total=h_total,
        e_inc=e_inc,
        h_inc=h_inc,
        e_sca_like=e_sca_like,
        h_sca_like=h_sca_like,
        e_total_mag=e_total_mag,
        h_total_mag=h_total_mag,
        e_sca_like_mag=e_sca_like_mag,
        h_sca_like_mag=h_sca_like_mag,
        s_total=s_total,
        s_inc=s_inc,
        s_sca_like=s_sca_like,
        s_delta=s_delta,
        s_total_mag=s_total_mag,
        s_inc_mag=s_inc_mag,
        s_sca_like_mag=s_sca_like_mag,
        s_delta_mag=s_delta_mag,
        plane_u_total=plane_u_total,
        plane_v_total=plane_v_total,
        plane_u_sca_like=plane_u_sca_like,
        plane_v_sca_like=plane_v_sca_like,
        plane_u_delta=plane_u_delta,
        plane_v_delta=plane_v_delta,
        extent_m=float(extent_m),
    )


# ============================================================
# Plotting helpers
# ============================================================

def plot_field_magnitude(
    field_result: FieldMapResult,
    *,
    quantity: FieldQuantity = "E",
    kind: Literal["total", "scattered", "delta"] = "total",
    scale: ScaleMode = "linear",
    floor: float = 1e-30,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    clip_percentile_low: float | None = None,
    clip_percentile_high: float | None = None,
    show_boundaries: bool = True,
    ax: plt.Axes | None = None,
):
    """
    Plot field or Poynting magnitude in the selected plane.

    Parameters
    ----------
    quantity :
        "E", "H", or "S"
    kind :
        - for "E" and "H": "total" or "scattered"
        - for "S": "total", "scattered", or "delta"
    scale :
        "linear" or "db"
    clip_percentile_low, clip_percentile_high :
        Optional percentile-based clipping for contrast control
    """
    data = _choose_magnitude_array(
        field_result,
        quantity=quantity,
        kind=kind,
    )

    data_plot = _apply_scale(data, scale=scale, floor=floor)

    if ax is None:
        fig, ax = plt.subplots()

    if clip_percentile_low is not None or clip_percentile_high is not None:
        vmin_p, vmax_p = _percentile_limits(
            data_plot,
            lower=clip_percentile_low,
            upper=clip_percentile_high,
        )
        if vmin is None:
            vmin = vmin_p
        if vmax is None:
            vmax = vmax_p

    extent_nm = [
        float(field_result.coord1_m[0] * 1e9),
        float(field_result.coord1_m[-1] * 1e9),
        float(field_result.coord2_m[0] * 1e9),
        float(field_result.coord2_m[-1] * 1e9),
    ]

    im = ax.imshow(
        data_plot,
        origin="lower",
        extent=extent_nm,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )

    if show_boundaries:
        _add_layer_boundaries(ax, field_result.radii_m)

    plane_labels = {
        "xy": ("x [nm]", "y [nm]"),
        "xz": ("x [nm]", "z [nm]"),
        "yz": ("y [nm]", "z [nm]"),
    }

    label_map = {
        ("E", "total"): r"|E| total",
        ("H", "total"): r"|H| total",
        ("S", "total"): r"|S| total",
        ("E", "scattered"): r"|E| scattered-like",
        ("H", "scattered"): r"|H| scattered-like",
        ("S", "scattered"): r"|S| scattered-like",
        ("S", "delta"): r"|S| delta",
    }
    label = label_map[(quantity, kind)]

    ax.set_xlabel(plane_labels[field_result.plane][0])
    ax.set_ylabel(plane_labels[field_result.plane][1])
    ax.set_title(
        f"{label} map at {field_result.wavelength_m * 1e9:.1f} nm ({field_result.plane} plane)"
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"{label} [dB]" if scale == "db" else label)

    return ax


def plot_poynting_streamlines(
    field_result: FieldMapResult,
    *,
    flow_kind: FlowKind = "total",
    background_quantity: FieldQuantity | Literal["none"] = "S",
    background_kind: Literal["total", "scattered", "delta"] = "total",
    background_scale: ScaleMode = "db",
    background_floor: float = 1e-30,
    background_cmap: str = "magma",
    background_clip_percentile_low: float | None = None,
    background_clip_percentile_high: float | None = 99.5,
    streamline_density: float = 1.2,
    streamline_color: str = "white",
    streamline_linewidth: float = 0.8,
    normalize_vectors: bool = False,
    min_speed_fraction: float = 0.01,
    mask_inside_sphere: bool = False,
    show_boundaries: bool = True,
    start_points_nm: np.ndarray | None = None,
    ax: plt.Axes | None = None,
):
    """
    Plot in-plane Poynting streamlines, optionally over a background map.

    Parameters
    ----------
    flow_kind :
        "total", "scattered", or "delta"
    background_quantity :
        "none", "E", "H", or "S"
    background_kind :
        - for E/H: "total" or "scattered"
        - for S  : "total", "scattered", or "delta"
    background_scale :
        "linear" or "db"
    normalize_vectors :
        If True, all non-zero vectors are normalized before streamplot.
        This often helps visibility of weak redistributed flow.
    min_speed_fraction :
        Vectors weaker than this fraction of the maximum in-plane speed
        are masked out to reduce jitter / broken streamline artefacts.
        Set to 0.0 to disable this masking.
    mask_inside_sphere :
        If True, suppress streamlines inside the outer sphere radius.
    start_points_nm :
        Optional streamline seed points in plotting coordinates [nm],
        shape (N, 2). If None, matplotlib chooses seeds automatically.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # --------------------------------------------------------
    # Background map
    # --------------------------------------------------------
    if background_quantity != "none":
        background_data = _choose_magnitude_array(
            field_result,
            quantity=background_quantity,
            kind=background_kind,
        )

        background_plot = _apply_scale(
            background_data,
            scale=background_scale,
            floor=background_floor,
        )

        if background_clip_percentile_low is not None or background_clip_percentile_high is not None:
            vmin, vmax = _percentile_limits(
                background_plot,
                lower=background_clip_percentile_low,
                upper=background_clip_percentile_high,
            )
        else:
            vmin = vmax = None

        extent_nm = [
            float(field_result.coord1_m[0] * 1e9),
            float(field_result.coord1_m[-1] * 1e9),
            float(field_result.coord2_m[0] * 1e9),
            float(field_result.coord2_m[-1] * 1e9),
        ]

        im = ax.imshow(
            background_plot,
            origin="lower",
            extent=extent_nm,
            cmap=background_cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        bg_label_map = {
            ("E", "total"): r"|E| total",
            ("H", "total"): r"|H| total",
            ("S", "total"): r"|S| total",
            ("E", "scattered"): r"|E| scattered-like",
            ("H", "scattered"): r"|H| scattered-like",
            ("S", "scattered"): r"|S| scattered-like",
            ("S", "delta"): r"|S| delta",
        }
        bg_label = bg_label_map[(background_quantity, background_kind)]

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f"{bg_label} [dB]" if background_scale == "db" else bg_label)

    # --------------------------------------------------------
    # Flow selection
    # --------------------------------------------------------
    if flow_kind == "total":
        U = np.asarray(field_result.plane_u_total, dtype=float)
        V = np.asarray(field_result.plane_v_total, dtype=float)
    elif flow_kind == "scattered":
        U = np.asarray(field_result.plane_u_sca_like, dtype=float)
        V = np.asarray(field_result.plane_v_sca_like, dtype=float)
    elif flow_kind == "delta":
        U = np.asarray(field_result.plane_u_delta, dtype=float)
        V = np.asarray(field_result.plane_v_delta, dtype=float)
    else:
        raise ValueError("flow_kind must be 'total', 'scattered', or 'delta'.")

    # --------------------------------------------------------
    # Masking
    # --------------------------------------------------------
    mask = np.zeros_like(U, dtype=bool)

    if mask_inside_sphere:
        mask |= field_result.radial_distance_m <= field_result.outer_radius_m

    speed = np.sqrt(U**2 + V**2)
    # Robust reference speed based on percentile, not absolute maximum
    ref_speed = _reference_speed_for_masking(
        speed,
        base_mask=mask,
        percentile=99.0,
    )

    if ref_speed > 0 and min_speed_fraction > 0:
        threshold = min_speed_fraction * ref_speed
        mask |= speed < threshold

    # Normalize if desired
    if normalize_vectors:
        speed = np.sqrt(U**2 + V**2)
        norm_mask = (~mask) & np.isfinite(speed) & (speed > 0)
        U_norm = np.zeros_like(U)
        V_norm = np.zeros_like(V)
        U_norm[norm_mask] = U[norm_mask] / speed[norm_mask]
        V_norm[norm_mask] = V[norm_mask] / speed[norm_mask]
        U = U_norm
        V = V_norm

    U_ma = np.ma.array(U, mask=mask)
    V_ma = np.ma.array(V, mask=mask)

    # --------------------------------------------------------
    # Streamline coordinates
    # --------------------------------------------------------
    X = field_result.coord1_m * 1e9
    Y = field_result.coord2_m * 1e9

    stream_kwargs = dict(
        x=X,
        y=Y,
        u=U_ma,
        v=V_ma,
        color=streamline_color,
        density=streamline_density,
        linewidth=streamline_linewidth,
        arrowsize=1.0,
    )

    if start_points_nm is not None:
        start_points_nm = np.asarray(start_points_nm, dtype=float)
        if start_points_nm.ndim != 2 or start_points_nm.shape[1] != 2:
            raise ValueError("start_points_nm must have shape (N, 2).")
        stream_kwargs["start_points"] = start_points_nm

    ax.streamplot(**stream_kwargs)

    if show_boundaries:
        _add_layer_boundaries(ax, field_result.radii_m)

    plane_labels = {
        "xy": ("x [nm]", "y [nm]"),
        "xz": ("x [nm]", "z [nm]"),
        "yz": ("y [nm]", "z [nm]"),
    }

    ax.set_xlabel(plane_labels[field_result.plane][0])
    ax.set_ylabel(plane_labels[field_result.plane][1])
    ax.set_title(
        f"Poynting streamlines ({flow_kind}) at {field_result.wavelength_m * 1e9:.1f} nm "
        f"({field_result.plane} plane)"
    )

    return ax

def plot_poynting_vectors(
    field_result: FieldMapResult,
    *,
    flow_kind: FlowKind = "delta",
    background_quantity: FieldQuantity | Literal["none"] = "S",
    background_kind: Literal["total", "scattered", "delta"] = "total",
    background_scale: ScaleMode = "db",
    background_floor: float = 1e-30,
    background_cmap: str = "magma",
    background_clip_percentile_low: float | None = None,
    background_clip_percentile_high: float | None = 99.5,
    step: int = 12,
    normalize_vectors: bool = True,
    min_speed_fraction: float = 0.01,
    mask_inside_sphere: bool = False,
    vector_color: str = "white",
    vector_scale: float | None = None,
    vector_width: float = 0.003,
    pivot: str = "mid",
    show_boundaries: bool = True,
    ax: plt.Axes | None = None,
):
    """
    Plot in-plane Poynting vectors as a quiver plot.

    This is often easier to interpret and debug than streamlines.

    Parameters
    ----------
    flow_kind :
        "total", "scattered", or "delta"
    background_quantity :
        "none", "E", "H", or "S"
    background_kind :
        - for E/H: "total" or "scattered"
        - for S  : "total", "scattered", or "delta"
    background_scale :
        "linear" or "db"
    step :
        Plot every `step`-th grid point in each direction
    normalize_vectors :
        If True, plot only vector directions (unit vectors for nonzero vectors)
    min_speed_fraction :
        Mask vectors whose speed is below this fraction of the max in-plane speed
    mask_inside_sphere :
        If True, suppress vectors inside the outer sphere radius
    vector_scale :
        Passed to matplotlib.quiver(scale=...). If None, matplotlib chooses automatically.
    pivot :
        Quiver pivot mode, e.g. "mid", "tail", or "tip"
    """
    if ax is None:
        fig, ax = plt.subplots()

    # --------------------------------------------------------
    # Optional background map
    # --------------------------------------------------------
    if background_quantity != "none":
        background_data = _choose_magnitude_array(
            field_result,
            quantity=background_quantity,
            kind=background_kind,
        )

        background_plot = _apply_scale(
            background_data,
            scale=background_scale,
            floor=background_floor,
        )

        if background_clip_percentile_low is not None or background_clip_percentile_high is not None:
            vmin, vmax = _percentile_limits(
                background_plot,
                lower=background_clip_percentile_low,
                upper=background_clip_percentile_high,
            )
        else:
            vmin = vmax = None

        extent_nm = [
            float(field_result.coord1_m[0] * 1e9),
            float(field_result.coord1_m[-1] * 1e9),
            float(field_result.coord2_m[0] * 1e9),
            float(field_result.coord2_m[-1] * 1e9),
        ]

        im = ax.imshow(
            background_plot,
            origin="lower",
            extent=extent_nm,
            cmap=background_cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        bg_label_map = {
            ("E", "total"): r"|E| total",
            ("H", "total"): r"|H| total",
            ("S", "total"): r"|S| total",
            ("E", "scattered"): r"|E| scattered-like",
            ("H", "scattered"): r"|H| scattered-like",
            ("S", "scattered"): r"|S| scattered-like",
            ("S", "delta"): r"|S| delta",
        }
        bg_label = bg_label_map[(background_quantity, background_kind)]

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f"{bg_label} [dB]" if background_scale == "db" else bg_label)

    # --------------------------------------------------------
    # Choose flow field
    # --------------------------------------------------------
    if flow_kind == "total":
        U = np.asarray(field_result.plane_u_total, dtype=float)
        V = np.asarray(field_result.plane_v_total, dtype=float)
    elif flow_kind == "scattered":
        U = np.asarray(field_result.plane_u_sca_like, dtype=float)
        V = np.asarray(field_result.plane_v_sca_like, dtype=float)
    elif flow_kind == "delta":
        U = np.asarray(field_result.plane_u_delta, dtype=float)
        V = np.asarray(field_result.plane_v_delta, dtype=float)
    else:
        raise ValueError("flow_kind must be 'total', 'scattered', or 'delta'.")

    speed = np.sqrt(U**2 + V**2)

    # --------------------------------------------------------
    # Masking
    # --------------------------------------------------------
    mask = np.zeros_like(U, dtype=bool)

    if mask_inside_sphere:
        mask |= field_result.radial_distance_m <= field_result.outer_radius_m

    # Robust reference speed based on percentile, not absolute maximum
    ref_speed = _reference_speed_for_masking(
        speed,
        base_mask=mask,
        percentile=99.0,
    )

    if ref_speed > 0 and min_speed_fraction > 0:
        mask |= speed < (min_speed_fraction * ref_speed)

    # --------------------------------------------------------
    # Normalize vectors if desired
    # --------------------------------------------------------
    if normalize_vectors:
        nonzero = (~mask) & np.isfinite(speed) & (speed > 0)
        U_plot = np.zeros_like(U)
        V_plot = np.zeros_like(V)
        U_plot[nonzero] = U[nonzero] / speed[nonzero]
        V_plot[nonzero] = V[nonzero] / speed[nonzero]
    else:
        U_plot = U.copy()
        V_plot = V.copy()

    # --------------------------------------------------------
    # Subsample grid
    # --------------------------------------------------------
    X_nm = field_result.grid1_m * 1e9
    Y_nm = field_result.grid2_m * 1e9

    Xq = X_nm[::step, ::step]
    Yq = Y_nm[::step, ::step]
    Uq = U_plot[::step, ::step]
    Vq = V_plot[::step, ::step]
    Mq = mask[::step, ::step]

    Uq = np.ma.array(Uq, mask=Mq)
    Vq = np.ma.array(Vq, mask=Mq)

    ax.quiver(
        Xq,
        Yq,
        Uq,
        Vq,
        color=vector_color,
        scale=vector_scale,
        width=vector_width,
        pivot=pivot,
    )

    if show_boundaries:
        _add_layer_boundaries(ax, field_result.radii_m)

    plane_labels = {
        "xy": ("x [nm]", "y [nm]"),
        "xz": ("x [nm]", "z [nm]"),
        "yz": ("y [nm]", "z [nm]"),
    }

    ax.set_xlabel(plane_labels[field_result.plane][0])
    ax.set_ylabel(plane_labels[field_result.plane][1])
    ax.set_title(
        f"Poynting vectors ({flow_kind}) at {field_result.wavelength_m * 1e9:.1f} nm "
        f"({field_result.plane} plane)"
    )

    return ax