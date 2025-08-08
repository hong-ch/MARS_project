#!/usr/bin/env python
#########################################
#       nii2png for Python 3.7          #
#         NIfTI Image Converter         #
#                v0.2.9                 #
#                                       #
#     Written by Alexander Laurence     #
# http://Celestial.Tokyo/~AlexLaurence/ #
#    alexander.adamlaurence@gmail.com   #
#              09 May 2019              #
#              MIT License              #
# Modified for axial view and batch      #
# processing by Grok                    #
#              07 Aug 2025              #
#########################################

import numpy, os, nibabel
import sys, getopt
from matplotlib import image

def main(argv):
    inputfile = ''
    outputdir = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('nii2png.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('nii2png.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-o", "--output"):
            outputdir = arg

    print('Input file is ', inputfile)
    print('Output folder is ', outputdir)

    # Load NIfTI file
    image_nib: nibabel.Nifti1Image = nibabel.load(inputfile)
    image_array = image_nib.get_fdata()
    print(f'Image dimensions: {image_array.shape}')

    # No rotation for batch processing
    print('Converting images without rotation.')

    # If 4D image inputted
    if len(image_array.shape) == 4:
        # Set 4D array dimension values
        nx, ny, nz, nw = image_array.shape

        # Set destination folder
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
            print("Created output directory: " + outputdir)

        print('Reading NIfTI file...')

        total_volumes = image_array.shape[3]
        total_slices = image_array.shape[0]  # Iterate over x-axis for axial view

        # Iterate through volumes
        for current_volume in range(0, total_volumes):
            slice_counter = 0
            # Iterate through slices (x-axis for axial view)
            for current_slice in range(0, total_slices):
                # Get axial slice
                data = image_array[current_slice, :, :, current_volume]
                            
                # Save as PNG
                print('Saving image...')
                image_name = os.path.basename(inputfile).split('.', 1)[0] + "_x" + "{:0>3}".format(str(current_slice+1)) + "_v" + "{:0>3}".format(str(current_volume+1)) + ".png"
                image.imsave(os.path.join(outputdir, image_name), data, cmap='gray')
                slice_counter += 1
                print('Saved.')

        print('Finished converting images')

    # Else if 3D image inputted
    elif len(image_array.shape) == 3:
        # Set 3D array dimension values
        nx, ny, nz = image_array.shape

        # Set destination folder
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
            print("Created output directory: " + outputdir)

        print('Reading NIfTI file...')

        total_slices = image_array.shape[0]  # Iterate over x-axis for axial view

        slice_counter = 0
        # Iterate through slices (x-axis for axial view)
        for current_slice in range(0, total_slices):
            # Get axial slice
            data = image_array[current_slice, :, :]

            # Save as PNG
            print('Saving image...')
            image_name = os.path.basename(inputfile).split('.', 1)[0] + "_x" + "{:0>3}".format(str(current_slice+1)) + ".png"
            image.imsave(os.path.join(outputdir, image_name), data, cmap='gray')
            slice_counter += 1
            print('Saved.')

        print('Finished converting images')
    else:
        print('Not a 3D or 4D Image. Please try again.')
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv[1:])