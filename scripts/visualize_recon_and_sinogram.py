import os
import h5py
import pydicom
import matplotlib.pyplot as plt

# 1) DICOM 불러와서 시각화
dcm_path = "C:/Users/user1/Desktop/Changhee/project/CT_to_radon/recon_ct_new.dcm"
ds       = pydicom.dcmread(dcm_path)
img      = ds.pixel_array     # pydicom이 자동으로 NumPy 배열로 변환해 줌

plt.figure(figsize=(6,6))
plt.title("Reconstructed CT (from DICOM)")
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()


# 2) HDF5 신호그램 불러와서 시각화
h5_path = "C:/Users/user1/Desktop/Changhee/project/CT_to_radon/sinogram.h5"
with h5py.File(h5_path, "r") as f:
    sino = f["sinogram"][()]   # 전체 배열 읽기

plt.figure(figsize=(8,4))
plt.title("Sinogram (from HDF5)")
plt.imshow(sino, aspect="auto", cmap="gray")
plt.xlabel("Projection angle index")
plt.ylabel("Detector bin index")
plt.show()
