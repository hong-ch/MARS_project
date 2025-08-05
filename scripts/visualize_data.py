import pydicom
import matplotlib.pyplot as plt
import os

def visualize_dicom(dicom_path, output_path):
    """Reads a DICOM file, visualizes it, and saves the visualization."""
    try:
        # Read the DICOM file
        dicom_image = pydicom.dcmread(dicom_path)

        # Plot the image
        plt.imshow(dicom_image.pixel_array, cmap=plt.cm.gray)
        plt.title(os.path.basename(dicom_path))
        plt.axis('off')

        # Save the plot
        plt.savefig(output_path)
        plt.close()
        print(f"Successfully visualized and saved to {output_path}")

    except Exception as e:
        print(f"Error processing {dicom_path}: {e}")

def find_first_dcm(start_path):
    """Recursively finds the first .dcm file in a directory."""
    for root, _, files in os.walk(start_path):
        for file in files:
            if file.endswith('.dcm'):
                return os.path.join(root, file)
    return None

if __name__ == "__main__":
    base_data_path = "C:/Users/user1/Desktop/Changhee/project/data"
    output_dir = "C:/Users/user1/Desktop/Changhee/project/visualizations"

    categories = {
        "Normal": "Normal",
        "PN": "PN",
        "TB": "TB"
    }

    for category_name, category_folder in categories.items():
        data_folder = os.path.join(base_data_path, category_folder)
        
        # Find the first patient folder
        patient_folders = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]
        if not patient_folders:
            print(f"No patient folders found in {data_folder}")
            continue
        
        first_patient_folder = os.path.join(data_folder, patient_folders[0])

        # Find the first DICOM file in the patient folder
        dicom_file_path = find_first_dcm(first_patient_folder)

        if dicom_file_path:
            output_filename = f"{category_name}_visualization.png"
            output_path = os.path.join(output_dir, output_filename)
            visualize_dicom(dicom_file_path, output_path)
        else:
            print(f"No DICOM files found in {first_patient_folder}")