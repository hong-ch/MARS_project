import os
import glob
import shutil
import random
import argparse
from sklearn.model_selection import train_test_split

def split_dataset(base_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits the NIfTI dataset into training, validation, and test sets.
    """
    if not os.path.exists(base_dir):
        print(f"오류: 입력 폴더 '{base_dir}'를 찾을 수 없습니다.")
        return

    if os.path.exists(output_dir):
        print(f"출력 폴더 '{output_dir}'가 이미 존재하여 삭제 후 다시 생성합니다.")
        shutil.rmtree(output_dir)

    print(f"출력 폴더 생성: {output_dir}")
    os.makedirs(output_dir)

    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    class_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not class_dirs:
        print(f"오류: 입력 폴더 '{base_dir}'에 클래스 하위 폴더(예: Normal, PN, TB)가 없습니다.")
        return
        
    print(f"발견된 클래스: {class_dirs}")

    for class_name in class_dirs:
        for split in splits:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

        class_path = os.path.join(base_dir, class_name)
        files = glob.glob(os.path.join(class_path, '*.nii.gz')) + glob.glob(os.path.join(class_path, '*.nii'))
        
        if not files:
            print(f"경고: {class_path} 에서 .nii 또는 .nii.gz 파일을 찾지 못했습니다.")
            continue

        # Ensure ratios sum to 1
        if (train_ratio + val_ratio + test_ratio) != 1.0:
            print("오류: 분할 비율의 합이 1이 되어야 합니다.")
            return

        # Split files
        train_files, test_val_files = train_test_split(files, test_size=(val_ratio + test_ratio), random_state=42)
        # Adjust test_size for the second split
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_files, test_files = train_test_split(test_val_files, test_size=relative_test_ratio, random_state=42)

        print(f"클래스 '{class_name}':")
        print(f"  - 전체 파일: {len(files)}개")
        print(f"  - 학습용: {len(train_files)}개")
        print(f"  - 검증용: {len(val_files)}개")
        print(f"  - 테스트용: {len(test_files)}개")

        # Copy files to destination
        for f in train_files:
            shutil.copy(f, os.path.join(output_dir, 'train', class_name))
        for f in val_files:
            shutil.copy(f, os.path.join(output_dir, 'val', class_name))
        for f in test_files:
            shutil.copy(f, os.path.join(output_dir, 'test', class_name))

    print("\n데이터셋 분할 완료.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NIfTI 데이터셋을 train, val, test로 분할합니다.")
    parser.add_argument('--input_dir', type=str, default='data/LDCT_nifti_test', help='클래스 폴더(Normal, PN, TB 등)를 포함하는 원본 데이터 폴더 경로')
    parser.add_argument('--output_dir', type=str, default='data/LDCT_nifti_split', help='분할된 데이터셋을 저장할 폴더 경로')
    args = parser.parse_args()
    
    split_dataset(args.input_dir, args.output_dir)
