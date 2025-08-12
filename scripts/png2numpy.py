import os
import numpy as np
from PIL import Image
import glob

base_dir = '../data/LDCT_PNG_test'
output_dir = '../data/LDCT_npy_test'
classes = {'Normal': 0, 'PN': 1, 'TB': 2}

os.makedirs(f'{output_dir}/volumes', exist_ok=True)
os.makedirs(f'{output_dir}/labels', exist_ok=True)

for cls in classes:
    cls_dir = f'{base_dir}/{cls}'
    patient_dirs = [d for d in os.listdir(cls_dir) if os.path.isdir(f'{cls_dir}/{d}')]

    for patient in patient_dirs:
        slices = sorted(glob.glob(f'{cls_dir}/{patient}/*.png'))  # z-axis 순서대로 정렬 (파일명에 따라)
        volume = []
        for slice_path in slices:
            img = Image.open(slice_path).convert('L')  # grayscale로 로드
            volume.append(np.array(img))
        volume = np.stack(volume, axis=0)  # (D, H, W) 형태의 3D array (D: depth/slice 수)
        volume = volume.astype(np.float32) / 255.0  # normalize (0~1 범위)

        # 전처리: 필요시 resize (e.g., (D, 128, 128)) 또는 intensity normalization (e.g., HU 값으로 변환, 하지만 PNG라 가정)
        # 예: from skimage.transform import resize
        # volume = resize(volume, (volume.shape[0], 128, 128), mode='constant')

        np.save(f'{output_dir}/volumes/{cls}_{patient}.npy', volume)
        with open(f'{output_dir}/labels/{cls}_{patient}.txt', 'w') as f:
            f.write(str(classes[cls]))