import os
import copy
import numpy as np
import SimpleITK as sitk
from skimage.transform import radon, iradon
import pydicom
from pydicom.encaps import encapsulate
from pydicom.uid import generate_uid
import h5py

# --- 설정 ---
dcm_template_path = (
    "C:/Users/user1/Desktop/Changhee/project/"
    "DICOM_data/Normal/000000098656/123199__0150_22189689.dcm"
)
out_dir = "C:/Users/user1/Desktop/Changhee/project/CT_to_radon"
os.makedirs(out_dir, exist_ok=True)

# --- 1) DICOM 읽어 CT 슬라이스 얻기 ---
image = sitk.ReadImage(dcm_template_path)
vol   = sitk.GetArrayFromImage(image)
CT    = vol[0, :, :] if vol.ndim == 3 else vol

# --- 2) Radon → Sinogram, Iradon → Reconstruction ---
theta    = np.linspace(0.0, 180.0, 180, endpoint=True)
sinogram = radon(CT, theta=theta, circle=False)
recon    = iradon(sinogram, circle=False, output_size=CT.shape[0])

# --- 3) Sinogram을 HDF5로 저장 ---
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

# --- 4) 새로운 DICOM 객체 복제 & UID 갱신 ---
orig_ds = pydicom.dcmread(dcm_template_path)

# deep copy 해서 메타데이터 원본 보존
new_ds = copy.deepcopy(orig_ds)

# 새 SOP Instance UID
new_sop_uid = generate_uid()
new_ds.SOPInstanceUID                          = new_sop_uid
new_ds.file_meta.MediaStorageSOPInstanceUID    = new_sop_uid

# 새 Series Instance UID (원한다면)
new_series_uid = generate_uid()
new_ds.SeriesInstanceUID = new_series_uid

# --- 5) Reconstruction 결과를 uint16로 변환 & 캡슐화 ---
recon_uint16 = np.clip(recon, 0, np.iinfo(np.uint16).max).astype(np.uint16)
frame_bytes    = [recon_uint16.tobytes()]
new_ds.PixelData = encapsulate(frame_bytes)
new_ds.Rows, new_ds.Columns = recon_uint16.shape

# (옵션) InstanceNumber 갱신
new_ds.InstanceNumber = 1

# --- 6) 새 DICOM 파일로 저장 ---
recon_dcm_path = os.path.join(out_dir, "recon_ct_new.dcm")
new_ds.save_as(recon_dcm_path, write_like_original=True)
print("New reconstructed CT saved as:", recon_dcm_path)
