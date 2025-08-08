import os
import glob
import subprocess
from pathlib import Path

def main():
    # Define project root and input/output directories
    project_root = os.getcwd()  # Assumes script is run from project root
    input_dir = os.path.join(project_root, "nifti", "TB")
    output_base_dir = os.path.join(project_root, "LDCT_PNG", "TB")

    # Ensure input directory exists
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist. Exiting...")
        return

    # Create output base directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)

    # Find all .nii.gz files in the input directory
    nii_files = glob.glob(os.path.join(input_dir, "*.nii.gz"))
    if not nii_files:
        print(f"No .nii.gz files found in {input_dir}. Exiting...")
        return

    print(f"Found {len(nii_files)} .nii.gz files in {input_dir}")

    # Process each .nii.gz file
    for nii_file in nii_files:
        # Extract the filename without extension for the output folder
        filename = os.path.splitext(os.path.splitext(os.path.basename(nii_file))[0])[0]
        output_dir = os.path.join(output_base_dir, filename)

        # Create patient-specific output directory
        os.makedirs(output_dir, exist_ok=True)
        print(f"Processing {nii_file} -> {output_dir}")

        # Call nii2png.py with the input file and output directory
        try:
            subprocess.run([
                "python", "nii2png.py",
                "-i", nii_file,
                "-o", output_dir
            ], check=True)
            print(f"Successfully converted {nii_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {nii_file}: {e}")

    print("Batch conversion completed.")

if __name__ == "__main__":
    main()