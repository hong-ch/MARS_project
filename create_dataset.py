
import os
import shutil
import pydicom

def get_slice_thickness(dicom_folder):
    """Reads the slice thickness from the first DICOM file in a folder."""
    files = [f for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
    if not files:
        return float('inf'), 0  # No DICOM files, return infinity thickness

    try:
        dicom_path = os.path.join(dicom_folder, files[0])
        dicom_dataset = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        thickness = float(dicom_dataset.get('SliceThickness', 'inf'))
        return thickness, len(files)
    except Exception:
        return float('inf'), len(files)

def create_project_dataset(source_dir, dest_dir):
    """Selects the higher-resolution folder and copies it to the destination."""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for category in ['Normal', 'PN', 'TB']:
        category_path = os.path.join(source_dir, category)
        dest_category_path = os.path.join(dest_dir, category)
        if not os.path.exists(dest_category_path):
            os.makedirs(dest_category_path)

        for patient_id in os.listdir(category_path):
            patient_path = os.path.join(category_path, patient_id)
            if not os.path.isdir(patient_path):
                continue

            subfolders = [os.path.join(patient_path, d) for d in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, d))]

            if len(subfolders) < 2:
                print(f"Skipping {patient_id} - not enough subfolders.")
                continue

            # Compare the two subfolders
            folder1, folder2 = subfolders[0], subfolders[1]
            thickness1, count1 = get_slice_thickness(folder1)
            thickness2, count2 = get_slice_thickness(folder2)

            # Choose the folder with smaller slice thickness, or more files if equal
            if thickness1 < thickness2:
                chosen_folder = folder1
            elif thickness2 < thickness1:
                chosen_folder = folder2
            elif count1 >= count2:
                chosen_folder = folder1
            else:
                chosen_folder = folder2

            # Copy the chosen folder's contents to the new dataset directory
            dest_patient_path = os.path.join(dest_category_path, patient_id)
            if not os.path.exists(dest_patient_path):
                 os.makedirs(dest_patient_path)

            for filename in os.listdir(chosen_folder):
                shutil.copy(os.path.join(chosen_folder, filename), dest_patient_path)
            
            print(f"Copied {os.path.basename(chosen_folder)} for patient {patient_id} to {dest_patient_path}")

if __name__ == "__main__":
    source_data_dir = "C:/Users/user1/Desktop/Changhee/project/data"
    destination_dataset_dir = "C:/Users/user1/Desktop/Changhee/project/project_dataset"
    create_project_dataset(source_data_dir, destination_dataset_dir)
