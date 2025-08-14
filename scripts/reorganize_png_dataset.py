
'''
This script reorganizes the ULDCT_png dataset by splitting it into
separate ''recon'' and ''sinogram'' directory trees.
'''
import os
import shutil
import sys

# --- Configuration ---
# The script assumes it is located in a subfolder (like ''scripts'')
# of your main project directory.
try:
    # Get the absolute path of the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to get the project root
    project_root = os.path.dirname(script_dir)
except NameError:
    # Fallback for interactive environments where __file__ is not defined
    project_root = os.path.abspath('.')

# Define the directory paths based on the project root
source_base_dir = os.path.join(project_root, 'data', 'ULDCT_png')
dest_sinogram_dir = os.path.join(project_root, 'data', 'ULDCT_png_sinogram')
dest_recon_dir = os.path.join(project_root, 'data', 'ULDCT_png_recon')
# ---------------------

def reorganize_files():
    '''
    Finds recon and sinogram png files in the source directory,
    moves them to the new directory structure, and cleans up the old directories.
    '''
    print("Starting file reorganization...")

    # 1. Create main destination directories
    print(f"Ensuring destination directory exists: {dest_sinogram_dir}")
    os.makedirs(dest_sinogram_dir, exist_ok=True)
    print(f"Ensuring destination directory exists: {dest_recon_dir}")
    os.makedirs(dest_recon_dir, exist_ok=True)

    # 2. Check if the source directory exists
    if not os.path.isdir(source_base_dir):
        print(f"Error: Source directory not found at '{source_base_dir}'")
        print("Please check the path in the script's Configuration section.")
        sys.exit(1)

    print(f"Scanning source: {source_base_dir}")
    print("-" * 50)

    # 3. Walk through the source directory structure
    # e.g., /data/ULDCT_png/{class_name}/{patient_id}/
    for class_name in os.listdir(source_base_dir):
        class_path = os.path.join(source_base_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"Processing class: {class_name}")

        for patient_id in os.listdir(class_path):
            patient_path = os.path.join(class_path, patient_id)
            if not os.path.isdir(patient_path):
                continue

            # --- Handle ''sinogram'' folder ---
            sinogram_source_folder = os.path.join(patient_path, 'sinogram')
            if os.path.isdir(sinogram_source_folder):
                # Define and create the new patient folder in the sinogram tree
                sinogram_dest_folder = os.path.join(dest_sinogram_dir, class_name, patient_id)
                os.makedirs(sinogram_dest_folder, exist_ok=True)

                # Move all files from source to destination
                for filename in os.listdir(sinogram_source_folder):
                    shutil.move(
                        os.path.join(sinogram_source_folder, filename),
                        os.path.join(sinogram_dest_folder, filename)
                    )

            # --- Handle ''recon'' folder ---
            recon_source_folder = os.path.join(patient_path, 'recon')
            if os.path.isdir(recon_source_folder):
                # Define and create the new patient folder in the recon tree
                recon_dest_folder = os.path.join(dest_recon_dir, class_name, patient_id)
                os.makedirs(recon_dest_folder, exist_ok=True)

                # Move all files from source to destination
                for filename in os.listdir(recon_source_folder):
                    shutil.move(
                        os.path.join(recon_source_folder, filename),
                        os.path.join(recon_dest_folder, filename)
                    )

    # 4. Clean up the old source directory
    print("\nAll files moved. Cleaning up the old directory structure...")
    try:
        shutil.rmtree(source_base_dir)
        print(f"Successfully removed old directory: {source_base_dir}")
    except OSError as e:
        print(f"Error during cleanup of '{source_base_dir}': {e}")
        print("You may need to remove it manually.")

    print("\nFile reorganization complete.")

if __name__ == '__main__':
    # This allows the script to be run directly from the command line
    reorganize_files()
