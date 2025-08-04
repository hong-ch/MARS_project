
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
import os
import pdb
import cv2
# Directories

CT=cv2.imread("C:/Users/user1/Desktop/Changhee/project/temp/CT_to_radon/CT123.jpg",0)
theta = np.linspace(0.0, 180.0, 90, endpoint=True)
sinogram = radon(CT, theta=theta, circle=False)
plt.imsave("C:/Users/user1/Desktop/Changhee/project/temp/CT_to_radon/sinogrma.png",sinogram,cmap='gray')
recon = iradon(sinogram)
plt.imsave("C:/Users/user1/Desktop/Changhee/project/temp/CT_to_radon/recon_ct.png",recon,cmap='gray')

