import os
import glob
import imageio
import numpy as np
import nibabel as nib
from tqdm import tqdm

def png_to_nifti(source_dir, dest_dir):
    """
    Converts PNG slices from the 'Normal' folder within ULDCT_png into NIfTI files.
    """
    print(f"Source directory: {source_dir}")
    print(f"Destination directory: {dest_dir}")

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")

    # Define the path to the 'Normal' folder
    normal_path = os.path.join(source_dir, 'Normal')
    dest_normal_path = os.path.join(dest_dir, 'Normal')

    if not os.path.exists(normal_path):
        print(f"Warning: 'Normal' folder not found in {source_dir}. Skipping.")
        return

    if not os.path.exists(dest_normal_path):
        os.makedirs(dest_normal_path)

    patient_folders = sorted([d for d in os.listdir(normal_path) if os.path.isdir(os.path.join(normal_path, d))])

    for patient_id in tqdm(patient_folders, desc="Processing patients in Normal", leave=False):
        patient_path = os.path.join(normal_path, patient_id)
        
        # Find PNG files and sort them numerically
        try:
            png_files = sorted(glob.glob(os.path.join(patient_path, 'recon', '*.png')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        except ValueError:
            # Handle cases where filenames are not purely numeric
            png_files = sorted(glob.glob(os.path.join(patient_path, 'recon', '*.png')))

        if not png_files:
            print(f"Warning: No PNG files found for {patient_id} in Normal. Skipping.")
            continue

        # Read all images for one patient
        volume_slices = []
        for png_file in png_files:
            try:
                slice_img = imageio.imread(png_file)
                if slice_img.ndim == 3:
                    # Convert RGB to grayscale if necessary
                    slice_img = slice_img.mean(axis=2)
                volume_slices.append(slice_img)
            except Exception as e:
                print(f"Error reading {png_file}: {e}")
        
        if not volume_slices:
            print(f"Warning: Could not read any valid PNG slices for {patient_id}. Skipping.")
            continue

        try:
            # Stack slices into a 3D volume
            # The slices are stacked along the 0 axis to match original format
            volume_3d = np.stack(volume_slices, axis=0).astype(np.float32)
            
            # Get dimensions
            dim_z, dim_y, dim_x = volume_3d.shape
            
            # Create an affine matrix with flips for x and y axes to correct orientations
            affine = np.eye(4)
            affine[0, 0] = -1  # Flip x (left-right)
            affine[0, 3] = dim_x - 1
            affine[1, 1] = -1  # Flip y (anterior-posterior / up-down in axial)
            affine[1, 3] = dim_y - 1
            # z remains unchanged

            # Create a NIfTI image object
            nifti_image = nib.Nifti1Image(volume_3d, affine)

            # Save the NIfTI file
            output_filename = os.path.join(dest_normal_path, f"{patient_id}.nii.gz")
            nib.save(nifti_image, output_filename)

        except Exception as e:
            print(f"Error creating NIfTI file for {patient_id}: {e}")

    print("\nConversion process completed.")

if __name__ == '__main__':
    # Define source and destination directories
    SOURCE_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ULDCT_png_Normal'))
    DEST_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ULDCT_nifti_Normal'))

    png_to_nifti(SOURCE_DATA_DIR, DEST_DATA_DIR)