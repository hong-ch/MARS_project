
import os

def rename_files():
    """
    Recursively renames files in a directory by removing '_sinogram' from their names.
    """
    # Set the root directory
    root_dir = os.path.join('data', 'ULDCT_png_sinogram')

    # Check if the directory exists
    if not os.path.isdir(root_dir):
        print(f"Error: Directory '{root_dir}' not found.")
        return

    # Walk through the directory
    for subdir, _, files in os.walk(root_dir):
        for filename in files:
            # Check if the file is a png and has '_sinogram' in its name
            if filename.endswith('_sinogram.png'):
                # Create the old and new file paths
                old_file_path = os.path.join(subdir, filename)
                new_filename = filename.replace('_sinogram.png', '.png')
                new_file_path = os.path.join(subdir, new_filename)

                # Rename the file
                try:
                    os.rename(old_file_path, new_file_path)
                    print(f'Renamed: {old_file_path} to {new_file_path}')
                except OSError as e:
                    print(f"Error renaming file {old_file_path}: {e}")

if __name__ == '__main__':
    rename_files()
