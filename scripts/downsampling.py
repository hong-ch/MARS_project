import os
import numpy as np
import SimpleITK as sitk
from skimage.transform import radon, iradon
import pydicom
from pydicom.encaps import encapsulate
import h5py

# --- 1) 경로 설정 및 출력 디렉터리 생성 ---
dcm_template_path = "C:/Users/user1/Desktop/Changhee/project/DICOM_data/Normal/000000098656/123199__0150_22189689.dcm"
out_dir = "C:/Users/user1/Desktop/Changhee/project/CT_to_radon"
os.makedirs(out_dir, exist_ok=True)

# --- 2) DICOM 읽어서 NumPy 배열로 변환 ---
image = sitk.ReadImage(dcm_template_path)
vol   = sitk.GetArrayFromImage(image)    # shape = (1, H, W) 또는 (H, W)
CT    = vol[0, :, :] if vol.ndim == 3 else vol

# --- 3) Radon 변환 (Sinogram 생성) ---
theta    = np.linspace(0.0, 180.0, 180, endpoint=True)
sinogram = radon(CT, theta=theta, circle=False)

# --- 4) 역변환 (Reconstruction) ---
recon = iradon(sinogram, circle=False, output_size=CT.shape[0])

# --- 5) Sinogram을 HDF5로 저장 ---
h5_path = os.path.join(out_dir, "sinogram.h5")
with h5py.File(h5_path, "w") as f:
    dset = f.create_dataset(
        "sinogram",
        data=sinogram.astype(np.float32),
        compression="gzip",
        compression_opts=4
    )
    dset.attrs["theta_start"] = float(theta[0])
    dset.attrs["theta_end"]   = float(theta[-1])
    dset.attrs["num_angles"]  = sinogram.shape[1]
print("Sinogram saved as HDF5:", h5_path)

# --- 6) Reconstructed CT를 원본 압축 Transfer Syntax 유지하며 DICOM으로 저장 ---
# 원본 헤더 불러오기
ds = pydicom.dcmread(dcm_template_path)

# 재구성 결과를 uint16 범위로 클리핑
recon_uint16 = np.clip(recon, 0, np.iinfo(np.uint16).max).astype(np.uint16)

# Frame bytes 생성 및 캡슐화
frame_bytes      = [recon_uint16.tobytes()]
encap_pixel_data = encapsulate(frame_bytes)

# PixelData 갱신 및 크기 정보 설정
ds.PixelData      = encap_pixel_data
ds.Rows, ds.Columns = recon_uint16.shape

# 파일 저장
recon_dcm_path = os.path.join(out_dir, "recon_ct.dcm")
ds.save_as(recon_dcm_path)
print("Reconstructed CT saved as compressed DICOM:", recon_dcm_path)
