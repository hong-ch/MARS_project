import h5py
import numpy as np

def print_hdf5_structure(name, obj):
    """Prints the name and type of groups and datasets in an HDF5 file."""
    print(name, end="")
    if isinstance(obj, h5py.Group):
        print(" (Group)")
    elif isinstance(obj, h5py.Dataset):
        print(f" (Dataset: shape {obj.shape}, dtype {obj.dtype})")
    else:
        print(" (Unknown object)")

# Path to the HDF5 file
file_path = 'temp_data/observation_challenge/observation_challenge_000.hdf5'

try:
    with h5py.File(file_path, 'r') as f:
        print(f"Contents of {file_path}:")
        f.visititems(print_hdf5_structure)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
