import numpy as np

_CIELAB_EPSILON = 216 / 24389  # ≈ 0.008856
_CIELAB_KAPPA = 24389 / 27     # ≈ 903.3

def angular_distance_deg(a, b):
    """Smallest absolute angular difference a↔b in degrees (0..180]."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)

def max_chroma_at_hue(Lab_ref, hue_deg, chroma, target_h, tol_deg=0.5):
    mask = angular_distance_deg(hue_deg, target_h) <= tol_deg
    if not np.any(mask):
        return None
    idxs = np.nonzero(mask)[0]
    i_best = idxs[np.argmax(chroma[mask])]
    L, a, b = Lab_ref[i_best]
    return dict(idx=int(i_best), L=float(L), a=float(a), b=float(b),
                C=float(chroma[i_best]), h=float(hue_deg[i_best]))

def chroma_envelope_vs_L_for_hue(Lab_ref, hue_deg, chroma, target_h,
                                  tol_deg=1.0, L_min=0.0, L_max=100.0, L_step=1.0):
    L_vals = Lab_ref[:, 0]
    bins = np.arange(L_min, L_max + L_step, L_step)
    L_centres, C_max_list, a_list, b_list = [], [], [], []
    hue_mask = angular_distance_deg(hue_deg, target_h) <= tol_deg
    if not np.any(hue_mask):
        return np.array([]), np.array([]), np.array([]), np.array([])
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = hue_mask & (L_vals >= lo) & (L_vals < hi)
        if np.any(m):
            idxs = np.nonzero(m)[0]
            imax = idxs[np.argmax(chroma[m])]
            L_centres.append(0.5 * (lo + hi))
            C_max_list.append(chroma[imax])
            a_list.append(Lab_ref[imax, 1])
            b_list.append(Lab_ref[imax, 2])
    return (np.asarray(L_centres), np.asarray(C_max_list),
            np.asarray(a_list), np.asarray(b_list))

def Lstar_to_Yrel(Lstar):
    L = np.asarray(Lstar, dtype=float)
    fy = (L + 16.0) / 116.0
    Yrel = np.where(L > (_CIELAB_KAPPA * _CIELAB_EPSILON), fy**3, L / _CIELAB_KAPPA)
    return Yrel

def max_chroma_stats_at_hue(Lab_ref, hue_deg, chroma, target_h, tol_deg=0.75):
    mask = angular_distance_deg(hue_deg, float(target_h)) <= float(tol_deg)
    if not np.any(mask):
        return None
    idxs = np.nonzero(mask)[0]
    chroma_scaled = 1.0 * chroma
    local_chroma = chroma_scaled[mask]
    best_local = idxs[np.argmax(local_chroma)]
    Lab_ref_scaled = 1.0 * Lab_ref
    L, a, b = Lab_ref_scaled[best_local]
    C = float(chroma_scaled[best_local])
    h = float(hue_deg[best_local])
    Y_rel = float(Lstar_to_Yrel(L))
    return dict(idx=int(best_local), h=h, C=C, L=float(L), a=float(a), b=float(b), Y_rel=Y_rel)

import plotly.graph_objects as go
import numpy as np

def add_hue_ridge_to_fig(fig, L_centres, a_vals, b_vals,
                          scale=1.0, name='Max-C* ridge at hue',
                          color='black'):
    if L_centres.size == 0:
        return fig
    fig.add_trace(go.Scatter3d(
        x=a_vals * scale,
        y=b_vals * scale,
        z=L_centres * scale,
        mode='lines+markers',
        name=name,
        line=dict(color=color, width=4),
        marker=dict(size=3, color=color),
        hovertemplate='L*: %{z:.2f}<br>a*: %{x:.2f}<br>b*: %{y:.2f}<extra></extra>'
    ))
    return fig

import pandas as pd
def max_chroma_per_hue(Rosch_DataFrame, hue_resolution = 1.0):
    """Compute the maximum chroma per hue from the Rosch-MacAdam DataFrame.
    Smooth the resulting a*, b*, L* values using Savitzky-Golay filter."""
    hue_bins = np.arange(0, 360 + hue_resolution, hue_resolution)  # 0 to 360 degrees

    # Prepare a list to store max chroma per hue bin
    max_chroma_per_hue = []

    # Loop through hue bins
    for i in range(len(hue_bins) - 1):
        hue_min = hue_bins[i]
        hue_max = hue_bins[i + 1]

        # Select points within the current hue bin
        bin_data = Rosch_DataFrame[(Rosch_DataFrame['hue_deg'] >= hue_min) & (Rosch_DataFrame['hue_deg'] < hue_max)]

        if len(bin_data) >= 1:
            # Keep only the maximum chroma per hue_deg to ensure strictly increasing x
            idx = bin_data['chroma'].idxmax()
            filtered_data = bin_data.loc[[idx]].reset_index(drop=True)
            sorted_data = filtered_data.sort_values(by='hue_deg')
            max_chroma_per_hue.append((sorted_data.iloc[0]['hue_deg'], sorted_data.iloc[0]['chroma'], sorted_data.iloc[0]['a'], sorted_data.iloc[0]['b'], sorted_data.iloc[0]['L'], sorted_data.iloc[0]['R'], sorted_data.iloc[0]['G'], sorted_data.iloc[0]['B']))
        elif len(bin_data) == 0:
            # No data in this bin, append NaNs
            max_chroma_per_hue.append((hue_min + hue_resolution / 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))

    # Convert result to DataFrame
    df_max_chroma = pd.DataFrame(max_chroma_per_hue, columns=['hue_deg', 'max_chroma', 'a', 'b', 'L', 'R', 'G', 'B'])

    from scipy.signal import savgol_filter
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import colour # type: ignore

    # Sort by hue to preserve circular structure
    df_sorted = df_max_chroma.sort_values(by='hue_deg').reset_index(drop=True)

    def cielab_to_hex(L, a, b):
        scale = 1
        L_arr = np.asarray(L)*scale
        a_arr = np.asarray(a)*scale
        b_arr = np.asarray(b)*scale
        # Build Lab array with shape (N, 3) for vectorized conversion or (3,) for scalar
        if L_arr.ndim == 0:
            Lab = np.array([float(L_arr), float(a_arr), float(b_arr)])
        else:
            Lab = np.column_stack((L_arr.astype(float), a_arr.astype(float), b_arr.astype(float)))

        XYZ = colour.Lab_to_XYZ(Lab)
        def rgb_to_hex(rgb):
            rgb8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
            return [f'#{r:02X}{g:02X}{b:02X}' for r, g, b in rgb8]
        sRGB = colour.XYZ_to_sRGB(XYZ)
        # Clip to [0, 1] range
        sRGB = np.clip(sRGB, 0, 1)
        hex_color = rgb_to_hex(sRGB)
        return hex_color

    # --- Robust handling for missing/insufficient data before smoothing ---
    # Interpolate missing values (linear along hue). This avoids NaNs being passed
    # to numpy.polyfit inside the savgol_filter implementation which triggers
    # SVD failures when NaNs are present or when there are too few valid points.
    for col in ['a', 'b', 'L']:
        # Use linear interpolation; fill ends by carrying nearest valid value if needed
        df_sorted[col] = df_sorted[col].interpolate(method='linear', limit_direction='both')

    # If interpolation left NaNs (e.g. all values were NaN), skip smoothing
    if df_sorted[['a', 'b', 'L']].isnull().any(axis=None):
        print("Warning: Not enough valid Rosch-MacAdam data to perform smoothing — skipping Savitzky-Golay.")
        df_sorted['a_smooth'] = df_sorted['a']
        df_sorted['b_smooth'] = df_sorted['b']
        df_sorted['L_smooth'] = df_sorted['L']
    else:
        # Define smoothing parameters
        window_length = 13  # Must be odd and < number of points
        polyorder = 4

        # Ensure window_length is an odd integer and smaller than available points
        n_points = len(df_sorted)
        if window_length >= n_points:
            # make window_length the largest odd < n_points
            window_length = n_points - 1 if n_points % 2 == 0 else n_points
            if window_length % 2 == 0:
                window_length -= 1

        # Ensure window_length > polyorder; if not, reduce polyorder
        if window_length <= polyorder:
            polyorder = max(1, window_length - 1)

        # Apply smoothing inside try/except to catch linear algebra failures
        try:
            df_sorted['a_smooth'] = savgol_filter(df_sorted['a'].to_numpy(dtype=float), window_length, polyorder)
            df_sorted['b_smooth'] = savgol_filter(df_sorted['b'].to_numpy(dtype=float), window_length, polyorder)
            df_sorted['L_smooth'] = savgol_filter(df_sorted['L'].to_numpy(dtype=float), window_length, polyorder)
        except Exception as e:
            print(f"Warning: Savitzky-Golay smoothing failed: {e}. Using unsmoothed data instead.")
            df_sorted['a_smooth'] = df_sorted['a']
            df_sorted['b_smooth'] = df_sorted['b']
            df_sorted['L_smooth'] = df_sorted['L']
        # Convert smoothed CIELAB to XYZ and then to sRGB and then to Hex
        df_sorted['hex_smooth'] = cielab_to_hex(df_sorted['L_smooth'], df_sorted['a_smooth'], df_sorted['b_smooth'])
    return df_sorted