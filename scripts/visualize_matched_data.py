
import pydicom
import matplotlib.pyplot as plt
import os

def visualize_matched_dicom(sinogram_path, recon_path, output_path):
    """Reads and visualizes matched sinogram and recon DICOM files."""
    try:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Read and display the sinogram image
        sinogram_image = pydicom.dcmread(sinogram_path)
        ax1.imshow(sinogram_image.pixel_array, cmap=plt.cm.gray)
        ax1.set_title(f"Sinogram: {os.path.basename(sinogram_path)}")
        ax1.axis('off')

        # Read and display the recon image
        recon_image = pydicom.dcmread(recon_path)
        ax2.imshow(recon_image.pixel_array, cmap=plt.cm.gray)
        ax2.set_title(f"Recon: {os.path.basename(recon_path)}")
        ax2.axis('off')

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Successfully visualized and saved to {output_path}")

    except Exception as e:
        print(f"Error processing files: {e}")

if __name__ == "__main__":
    base_patient_path = "C:/Users/user1/Desktop/Changhee/project/data/TB/000000098893"
    output_dir = "C:/Users/user1/Desktop/Changhee/project/visualizations"

    sinogram_folder = os.path.join(base_patient_path, "2. helical")
    recon_folder = os.path.join(base_patient_path, "3. recon")

    # Get the first file from each folder for visualization
    sinogram_files = sorted(os.listdir(sinogram_folder))
    recon_files = sorted(os.listdir(recon_folder))

    if sinogram_files and recon_files:
        first_sinogram = os.path.join(sinogram_folder, sinogram_files[0])
        first_recon = os.path.join(recon_folder, recon_files[0])

        output_filename = "TB_matched_visualization.png"
        output_path = os.path.join(output_dir, output_filename)

        visualize_matched_dicom(first_sinogram, first_recon, output_path)
    else:
        print("Could not find sinogram or recon files to visualize.")
