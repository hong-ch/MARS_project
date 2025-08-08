
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
import os
import pdb
import cv2
# Directories

CT=cv2.imread("C:/Users/user1/Desktop/Changhee/project/LDCT_PNG/TB/000000098893/000000098893_x130.png",0)

# radon 변환
theta = np.linspace(0.0, 180.0, 90, endpoint=True)
sinogram = radon(CT, theta=theta, circle=False)
x = CT.shape[0]
y = CT.shape[1]
plt.imsave("C:/Users/user1/Desktop/Changhee/project/temp/CT_to_radon/sinogram.png",sinogram,cmap='gray')

# iradon 및 저장
recon = iradon(sinogram,circle=False,output_size=x)
plt.imsave("C:/Users/user1/Desktop/Changhee/project/temp/CT_to_radon/recon_ct.png",recon,cmap='gray')
