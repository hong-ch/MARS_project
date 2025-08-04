

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
import pydicom
import os

# --- 사용자 설정 영역 시작 ---

# TODO: 여기에 실제 DICOM 파일의 전체 경로를 입력하세요.
# 예: dicom_file_path = "C:/Users/user1/Desktop/Changhee/project/dicom_data/your_file.dcm"
dicom_file_path = "path/to/your/dicom_file.dcm"  # <--- 이 부분을 수정하세요.

# 결과를 저장할 폴더 경로
output_dir = "reconstructed_from_dicom"

# Radon 변환에 사용할 투영 각도 수
num_projections = 180

# --- 사용자 설정 영역 끝 ---

# 출력 폴더 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# DICOM 파일 읽기
try:
    # pydicom을 사용하여 DICOM 파일을 읽습니다.
    dcm_data = pydicom.dcmread(dicom_file_path)

    # DICOM 파일에서 실제 픽셀 데이터(NumPy 배열)를 추출합니다.
    ct_image = dcm_data.pixel_array

    print(f"성공: DICOM 파일을 읽었습니다. 이미지 크기: {ct_image.shape}")

    # Radon 변환을 수행하여 사이노그램을 생성합니다.
    # 0도부터 180도까지 지정된 수의 각도로 투영 데이터를 생성합니다.
    theta = np.linspace(0.0, 180.0, num_projections, endpoint=False)
    sinogram = radon(ct_image, theta=theta, circle=True)

    # 생성된 사이노그램을 이미지 파일로 저장합니다.
    sinogram_path = os.path.join(output_dir, "generated_sinogram.png")
    plt.imsave(sinogram_path, sinogram, cmap='gray')
    print(f"사이노그램이 여기에 저장되었습니다: {sinogram_path}")

    # 역 Radon 변환(Filtered Back-Projection)을 수행하여 이미지를 재구성합니다.
    reconstructed_image = iradon(sinogram, theta=theta, circle=True)

    # 재구성된 이미지를 파일로 저장합니다.
    reconstruction_path = os.path.join(output_dir, "reconstructed_image_from_dicom.png")
    plt.imsave(reconstruction_path, reconstructed_image, cmap='gray')
    print(f"재구성된 이미지가 여기에 저장되었습니다: {reconstruction_path}")

except FileNotFoundError:
    print(f"오류: 지정된 경로에 파일이 없습니다. \n'{dicom_file_path}'\n경로를 다시 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")
    print("DICOM 파일이 맞는지, 또는 파일이 손상되지 않았는지 확인해주세요.")

