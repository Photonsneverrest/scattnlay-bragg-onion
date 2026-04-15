import numpy as np

def save_colour_solid_to_csv(filename, XYZ_rel, Lab_ref, rgb_bright, hue_deg, chroma):
    """
    Save the Rösch–MacAdam colour solid data to a CSV file.

    Parameters:
    - filename: str, path to the output CSV file.
    - XYZ_rel: (N, 3) array of relative XYZ values.
    - Lab_ref: (N, 3) array of CIELAB values.
    - rgb_bright: (N, 3) array of brightened sRGB values.
    - hue_deg: (N,) array of hue angles in degrees.
    - chroma: (N,) array of chroma values.
    """
    data = np.column_stack([XYZ_rel, Lab_ref, rgb_bright, hue_deg, chroma])
    header = "X,Y,Z,L,a,b,R,G,B,hue_deg,chroma"
    np.savetxt(filename, data, delimiter=",", header=header, comments='')

def load_colour_solid_from_csv(filename):
    """
    Load the Rösch–MacAdam colour solid data from a CSV file.

    Parameters:
    - filename: str, path to the input CSV file.

    Returns:
    - XYZ_rel: (N, 3) array of relative XYZ values.
    - Lab_ref: (N, 3) array of CIELAB values.
    - rgb_bright: (N, 3) array of brightened sRGB values.
    - hue_deg: (N,) array of hue angles in degrees.
    - chroma: (N,) array of chroma values.
    """
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    XYZ_rel = data[:, 0:3]
    Lab_ref = data[:, 3:6]
    rgb_bright = data[:, 6:9]
    hue_deg = data[:, 9]
    chroma = data[:, 10]
    return XYZ_rel, Lab_ref, rgb_bright, hue_deg, chroma