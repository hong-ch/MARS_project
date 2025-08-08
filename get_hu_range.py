import os
import pydicom
import numpy as np
import sys

def get_hu_range_for_dataset(directory):
    """
    Calculates the minimum and maximum Hounsfield Unit (HU) values for all DICOM files
    in a directory and its subdirectories.

    Args:
        directory (str): The path to the directory containing DICOM files.

    Returns:
        tuple: A tuple containing the global minimum and maximum HU values found.
               Returns (None, None) if no DICOM files are found.
    """
    global_min_hu = float('inf')
    global_max_hu = float('-inf')
    dicom_files_found = False
    file_count = 0

    # First, count the total number of files to process for progress indication
    total_files = sum([len(files) for r, d, files in os.walk(directory)])
    
    print(f"Searching for DICOM files in: {directory}")
    print(f"Found {total_files} total files to check.")

    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_count += 1
            
            # Print progress
            progress = (file_count / total_files) * 100
            sys.stdout.write(f"\rProcessing file {file_count}/{total_files} ({progress:.2f}%) - {os.path.basename(filepath)}")
            sys.stdout.flush()

            try:
                # Read the DICOM file, deferring pixel data loading for efficiency
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)

                # Check if it's a CT image by looking for Rescale Slope/Intercept
                if not hasattr(dcm, 'RescaleSlope') or not hasattr(dcm, 'RescaleIntercept'):
                    continue
                
                # Now read the full file since it's likely a valid CT
                dcm = pydicom.dcmread(filepath, force=True)

                # Get the pixel array
                image = dcm.pixel_array

                # Convert to HU
                rescale_slope = dcm.RescaleSlope
                rescale_intercept = dcm.RescaleIntercept
                
                # Ensure calculations are done with floating point numbers
                hu_image = image.astype(np.float64) * float(rescale_slope) + float(rescale_intercept)

                # Update global min and max
                min_hu = np.min(hu_image)
                max_hu = np.max(hu_image)

                if min_hu < global_min_hu:
                    global_min_hu = min_hu
                if max_hu > global_max_hu:
                    global_max_hu = max_hu

                dicom_files_found = True

            except Exception:
                # Ignore files that are not valid DICOM files or cause other errors
                continue

    sys.stdout.write("\n") # Move to the next line after progress bar
    
    if not dicom_files_found:
        return None, None

    return global_min_hu, global_max_hu

if __name__ == '__main__':
    # The script is expected to be in the project root, so 'data_DICOM' is a subdirectory.
    dicom_directory = 'data_DICOM'
    
    if not os.path.isdir(dicom_directory):
        print(f"Error: Directory '{dicom_directory}' not found.")
        print("Please make sure the script is in the correct project directory and the 'data_DICOM' folder exists.")
    else:
        min_val, max_val = get_hu_range_for_dataset(dicom_directory)

        if min_val is not None and max_val is not None:
            print("\n--- Overall HU Range for the Dataset ---")
            print(f"Minimum HU value: {min_val}")
            print(f"Maximum HU value: {max_val}")
            print("----------------------------------------")
        else:
            print("\nNo valid DICOM files with HU information were found in the specified directory.")
