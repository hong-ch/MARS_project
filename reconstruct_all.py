import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import iradon
import os

# Directories
data_dir = 'temp_data/observation_challenge'
output_dir = 'reconstructions'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all the files from 000 to 028
for i in range(29):
    file_name = f'observation_challenge_{i:03d}.hdf5'
    file_path = os.path.join(data_dir, file_name)
    output_image_name = f'reconstruction_{i:03d}.png'
    output_image_path = os.path.join(output_dir, output_image_name)

    try:
        with h5py.File(file_path, 'r') as f:
            # The sinogram seems to be stored as (projections, detectors)
            sinogram_raw = f['data'][0, :, :].astype(np.float64)

        # Transpose the sinogram to (detectors, projections) for iradon
        sinogram = sinogram_raw.T

        # Define the projection angles
        # The sinogram has 1000 projections, spread over 180 degrees
        theta = np.linspace(0., 180., sinogram_raw.shape[0], endpoint=False)

        # Perform the inverse Radon transform (Filtered Back-Projection)
        reconstructed_image = iradon(sinogram, theta=theta, circle=True)

        # Save the reconstructed image
        plt.imsave(output_image_path, reconstructed_image, cmap='gray')
        print(f"Reconstruction saved to {output_image_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred with {file_name}: {e}")