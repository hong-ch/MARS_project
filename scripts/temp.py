import os
import numpy as np
import SimpleITK as sitk
from skimage.transform import radon, iradon
import matplotlib.pyplot as plt

# 0) 출력 폴더 설정 및 생성
out_dir = r"C:/Users/user1/Desktop/Changhee/project/CT_to_radon"
os.makedirs(out_dir, exist_ok=True)

# 1) NIfTI 파일 읽기
nii_path = r"C:/Users/user1/Desktop/Changhee/project/data/Normal/000000098783.nii.gz"
image    = sitk.ReadImage(nii_path)
vol      = sitk.GetArrayFromImage(image)   # shape = (Z, Y, X)

# 2) 원하는 슬라이스 선택 (예: 중앙 Axial 슬라이스)
z_mid    = vol.shape[2] // 2
slice2d  = vol[ :, :, z_mid]               # (Y, X) 형태의 2D 배열

# 3) Radon 변환
theta    = np.linspace(0.0, 180.0, 90, endpoint=True)
sinogram = radon(slice2d, theta=theta, circle=False)

# 4) Iradon 역변환
recon    = iradon(sinogram, theta=theta, circle=False, output_size=slice2d.shape[0])

# 5) 결과 저장
plt.imsave(os.path.join(out_dir, "temp_original_slice.png"), slice2d, cmap="gray")
plt.imsave(os.path.join(out_dir, "temp_sinogram.png"),       sinogram, cmap="gray")
plt.imsave(os.path.join(out_dir, "temp_reconstruction.png"), recon,    cmap="gray")

# 6) (선택) 동시에 화면에도 출력
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))

ax1.set_title("Axial Slice");          ax1.imshow(slice2d, cmap="gray");  ax1.axis("off")
ax2.set_title("Sinogram");             ax2.imshow(sinogram, aspect="auto", cmap="gray"); ax2.axis("off")
ax3.set_title("Reconstruction");       ax3.imshow(recon, cmap="gray");        ax3.axis("off")

plt.tight_layout()
plt.show()
