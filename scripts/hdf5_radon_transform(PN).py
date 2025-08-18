import numpy as np
import os
import glob
import cv2
import h5py
from skimage.transform import radon

# Base directory for PNG data
base_png_dir = "C:/Users/user1/Desktop/Changhee/project/data/LDCT_png_image"
# Base directory for the new dataset
new_dataset_dir = "C:/Users/user1/Desktop/Changhee/project/data/ULDCT_hdf5_sinogram"

# Create the new dataset directory if it doesn't exist
os.makedirs(new_dataset_dir, exist_ok=True)

# Define the specific sub-directory to process
sub_dir = "PN"
sub_dir_path = os.path.join(base_png_dir, sub_dir)
if os.path.isdir(sub_dir_path):
    # Iterate over each patient folder in the "PN" directory
    for patient_dir in os.listdir(sub_dir_path):
        patient_dir_path = os.path.join(sub_dir_path, patient_dir)
        if os.path.isdir(patient_dir_path):
            # Find all PNG files in the patient folder
            png_files = sorted(glob.glob(os.path.join(patient_dir_path, "*.png")))  # Sort for consistent ordering
            if png_files:
                # Calculate the middle index and select 100 slices (50 before and 50 after)
                total_slices = len(png_files)
                middle_idx = total_slices // 2
                start_idx = max(0, middle_idx - 50)  # Ensure no negative index
                end_idx = min(total_slices, middle_idx + 50)  # Ensure no out-of-range index
                selected_png_files = png_files[start_idx:end_idx]

                if len(selected_png_files) >= 100:  # Ensure at least 100 slices
                    # Use the patient directory directly without "sinogram" subfolder
                    new_patient_dir = os.path.join(new_dataset_dir, sub_dir, patient_dir)
                    os.makedirs(new_patient_dir, exist_ok=True)

                    for png_path in selected_png_files:
                        # Read the PNG file in grayscale
                        CT = cv2.imread(png_path, 0)
                        if CT is None:
                            print(f"Failed to load {png_path}. Skipping...")
                            continue

                        # Get the original filename without extension
                        base_filename = os.path.splitext(os.path.basename(png_path))[0]

                        # Radon transform with increased angles
                        theta = np.linspace(0.0, 180.0, 90, endpoint=False)  
                        sinogram = radon(CT, theta=theta, circle=False)  
                        sinogram = sinogram.astype(np.float32)  # Ensure float32

                        # Save as HDF5 directly in patient folder
                        sinogram_filename = os.path.join(new_patient_dir, f"{base_filename}_sinogram.h5")
                        with h5py.File(sinogram_filename, "w") as f:
                            f.create_dataset("sinogram", data=sinogram)

print("Sinogram creation for 'PN' directory complete.")