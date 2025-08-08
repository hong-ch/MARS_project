import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
import SimpleITK as sitk
import os
import pdb
# 1) DICOM 읽기
dcm_path = "C:/Users/user1/Desktop/Changhee/project/DICOM_data/Normal/000000098656/123199__0150_22189689.dcm"
image   = sitk.ReadImage(dcm_path)
vol     = sitk.GetArrayFromImage(image)   # shape = (1, H, W) 또는 (H, W)


# 2) 2D 배열로 변환
if vol.ndim == 3:
    CT = vol[0, :, :]                     # Single-slice DICOM 이므로 첫 번째 슬라이스 사용
else:
    CT = vol

plt.imsave("C:/Users/user1/Desktop/Changhee/project/CT_to_radon/original_ct.png",CT,cmap='gray')

# 3) radon 변환
theta = np.linspace(0.0, 180.0, 90, endpoint=True)
sinogram = radon(CT, theta=theta, circle=False)
x = CT.shape[0]
y = CT.shape[1]
plt.imsave("C:/Users/user1/Desktop/Changhee/project/CT_to_radon/sinogram.png",sinogram,cmap='gray')

# 4) iradon 및 저장
recon = iradon(sinogram,circle=False,output_size=x)
plt.imsave("C:/Users/user1/Desktop/Changhee/project/CT_to_radon/recon_ct.png",recon,cmap='gray')

