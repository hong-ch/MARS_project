

import pydicom
import os

def read_metadata(file_path):
    """Reads and prints the metadata of a DICOM file."""
    try:
        dicom_dataset = pydicom.dcmread(file_path)
        print(f"--- Metadata for: {os.path.basename(file_path)} ---")
        print(dicom_dataset)
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    sinogram_file = "C:/Users/user1/Desktop/Changhee/project/data/TB/000000098893/2. helical/123163_helical_0001_22168687.dcm"
    recon_file = "C:/Users/user1/Desktop/Changhee/project/data/TB/000000098893/3. recon/123163_recon_0001_22172719.dcm"

    read_metadata(sinogram_file)
    read_metadata(recon_file)

