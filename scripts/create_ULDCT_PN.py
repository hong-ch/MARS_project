import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
import os
import glob
import cv2

# Base directory for PNG data
base_png_dir = "C:/Users/user1/Desktop/Changhee/project/data/LDCT_png"
# Base directory for the new dataset
new_dataset_dir = "C:/Users/user1/Desktop/Changhee/project/data/ULDCT_png"

# Create the new dataset directory if it doesn't exist
os.makedirs(new_dataset_dir, exist_ok=True)

# Process only the PN sub-directory
sub_dir = "PN"
sub_dir_path = os.path.join(base_png_dir, sub_dir)
if os.path.isdir(sub_dir_path):
    # Iterate over each patient folder in PN
    for patient_dir in os.listdir(sub_dir_path):
        patient_dir_path = os.path.join(sub_dir_path, patient_dir)
        if os.path.isdir(patient_dir_path):
            # Find all PNG files in the patient folder
            png_files = glob.glob(os.path.join(patient_dir_path, "*.png"))
            if png_files:
                # Create patient-specific directories in the new dataset
                new_patient_dir_sinogram = os.path.join(new_dataset_dir, sub_dir, patient_dir, "sinogram")
                new_patient_dir_image = os.path.join(new_dataset_dir, sub_dir, patient_dir, "recon")
                os.makedirs(new_patient_dir_sinogram, exist_ok=True)
                os.makedirs(new_patient_dir_image, exist_ok=True)

                for png_path in png_files:
                    # Read the PNG file in grayscale
                    CT = cv2.imread(png_path, 0)
                    if CT is None:
                        print(f"Failed to load {png_path}. Skipping...")
                        continue

                    # Get the original filename without extension
                    base_filename = os.path.splitext(os.path.basename(png_path))[0]

                    # Radon transform
                    theta = np.linspace(0.0, 180.0, 90, endpoint=True)
                    sinogram = radon(CT, theta=theta, circle=False)
                    sinogram_filename = os.path.join(new_patient_dir_sinogram, f"{base_filename}_sinogram.png")
                    plt.imsave(sinogram_filename, sinogram, cmap='gray')

                    # Inverse Radon transform
                    recon = iradon(sinogram, circle=False, output_size=CT.shape[0])
                    recon_filename = os.path.join(new_patient_dir_image, f"{base_filename}_recon.png")
                    plt.imsave(recon_filename, recon, cmap='gray')

print("Dataset creation for PN folder complete.")