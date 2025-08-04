
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

# Directory containing the HDF5 files
data_dir = 'temp_data/observation_challenge'
output_dir = 'visualizations'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all the files from 000 to 028
for i in range(29):
    file_name = f'observation_challenge_{i:03d}.hdf5'
    file_path = os.path.join(data_dir, file_name)
    output_image_name = f'visualization_{i:03d}.png'
    output_image_path = os.path.join(output_dir, output_image_name)

    try:
        with h5py.File(file_path, 'r') as f:
            data = f['data']
            
            if data.ndim >= 2:
                if data.ndim == 3:
                    slice_2d = data[0, :, :]
                else:
                    slice_2d = data[:,:]

                plt.figure(figsize=(10, 8))
                plt.imshow(slice_2d, aspect='auto', cmap='viridis')
                plt.colorbar(label='Intensity')
                plt.title(f'Visualization of {file_name}')
                plt.xlabel('X-axis')
                plt.ylabel('Y-axis')
                
                plt.savefig(output_image_path)
                plt.close() # Close the figure to free up memory
                print(f"Visualization saved to {output_image_path}")
            else:
                print(f"Data in {file_path} is not suitable for image visualization.")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred with {file_name}: {e}")
