import os
import nibabel as nib
import numpy as np
import scipy.ndimage.interpolation import zoom
import scipy as sp
from tqdm.notebook import tqdm_notebook

def load_nifti(file_path, mask=None, z_factor=None, remove_nan=False):
    """Load a 3D array from a NIFTI file."""
    img = nib.load(file_path)
    struct_arr = np.array(img.get_data())

    if remove_nan:
        struct_arr = np.nan_to_num(struct_arr)
    if mask is not None:
        struct_arr *= mask
    if z_factor is not None:
        struct_arr = np.around(zoom(struct_arr, z_factor), 0)

    return struct_arr


def save_nifti(file_path, struct_arr):
    """Save a 3D array to a NIFTI file."""
    img = nib.Nifti1Image(struct_arr, np.eye(4))
    nib.save(img, file_path)
