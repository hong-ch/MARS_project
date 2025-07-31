import h5py
import matplotlib.pyplot as plt
import numpy as np

# Path to the HDF5 file
file_path = 'temp_data/observation_challenge/observation_challenge_010.hdf5'
output_image_path = 'visualization_010.png'

try:
    with h5py.File(file_path, 'r') as f:
        # Access the dataset
        data = f['data']
        
        # Check if the data has at least 3 dimensions
        if data.ndim >= 2:
            # Select a 2D slice to visualize. For 3D data, we take the first slice.
            if data.ndim == 3:
                slice_2d = data[0, :, :]
            else: # Assumes 2D data
                slice_2d = data[:,:]

            # Create a plot
            plt.figure(figsize=(10, 8))
            plt.imshow(slice_2d, aspect='auto', cmap='viridis')
            plt.colorbar(label='Intensity')
            plt.title('Visualization of a 2D Slice from HDF5 Data')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            
            # Save the plot to a file
            plt.savefig(output_image_path)
            print(f"Visualization saved to {output_image_path}")
        else:
            print(f"Data in {file_path} is not 2D or 3D, cannot visualize as an image.")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
