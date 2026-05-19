# scattering_extract.py

import numpy as np

from bragg_onion.materials import ConstantDispersion
from bragg_onion.sweep import run_bragg_onion_sweep


def rerun_single_case(row_case, wavelengths_nm, theta_deg):
    mat_A = ConstantDispersion.from_nk("A", n=float(row_case["n_A"]), k=float(row_case["k"]))
    mat_B = ConstantDispersion.from_nk(
        "B",
        n=float(row_case.get("n_B", 1.5)),
        k=float(row_case["k"]),
    )

    single_case_grid = {
        "peak_wavelength_m": [float(row_case["design_peak_wavelength_nm"]) * 1e-9],
        "n_layers": [int(row_case["n_layers"])],
        "outer_layer": [row_case["outer_layer"]],
        "core_thickness_factor": [float(row_case.get("core_thickness_factor", 0.5))],
    }

    collection_na = float(row_case.get(
        "collection_NA",
        np.sin(np.deg2rad(row_case["collection_angle_deg"]))
    ))

    return run_bragg_onion_sweep(
        geometry_mode="peak_wavelength",
        parameter_grid=single_case_grid,
        material_a=mat_A,
        material_b=mat_B,
        n_medium=1.0,
        wavelengths_m=wavelengths_nm * 1e-9,
        theta_rad=np.deg2rad(theta_deg),
        collection_na=collection_na,
        collection_direction="backward",
        integration_quantity_for_colour="q_collected_geom",
        store_full_results=True,
        progress=False,
    )


def extract_case_arrays(sweep_case):
    cr = sweep_case.case_results[0]

    scat = cr.scattering_result
    integ = cr.integrated_result
    col = cr.colour_result

    return {
        # grids
        "wavelengths_nm": np.asarray(scat.wavelengths_m) * 1e9,
        "theta_deg": np.rad2deg(np.asarray(scat.theta_rad)),

        # scattering
        "qsca": np.asarray(scat.qsca),
        "qext": np.asarray(scat.qext),
        "qabs": np.asarray(scat.qabs),
        "qbk": np.asarray(scat.qbk),
        "g": np.asarray(scat.g),
        "albedo": np.asarray(scat.albedo),
        "dcs_geom": np.asarray(scat.dcs_geom_norm_sr_inv),

        # integrated
        "wavelengths_int_nm": np.asarray(integ.wavelengths_m) * 1e9,
        "q_collected_geom": np.asarray(integ.q_collected_geom),
        "fraction_collected": np.asarray(integ.fraction_collected),

        # colour
        "wavelengths_colour_nm": np.asarray(col.wavelengths_nm),
        "spectrum_used": np.asarray(col.spectrum_used),
    }
