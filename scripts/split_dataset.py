'''
데이터셋을 train, validation, test 세트로 분할합니다.
'''
import os
import glob
import random
import shutil

def split_data():
    # 기본 경로 설정
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_dir = os.path.join(base_dir, 'data', 'LDCT_png_test')
    output_dir = os.path.join(base_dir, 'data', 'LDCT_png_split')

    # 클래스 및 분할 비율 정의
    classes = ['Normal', 'PN', 'TB']
    split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}

    # 출력 폴더 생성
    if os.path.exists(output_dir):
        print(f"출력 폴더 {output_dir}가 이미 존재합니다. 기존 폴더를 삭제하고 다시 시작합니다.")
        shutil.rmtree(output_dir)
    
    for split_name in split_ratios.keys():
        for class_name in classes:
            os.makedirs(os.path.join(output_dir, split_name, class_name), exist_ok=True)
    
    print(f"입력 폴더: {input_dir}")
    print(f"출력 폴더: {output_dir}")
    print("-" * 30)

    # 각 클래스에 대해 파일 분할 수행
    for class_name in classes:
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            print(f"경고: {class_path} 폴더를 찾을 수 없습니다. 이 클래스는 건너뜁니다.")
            continue

        # PNG 파일 목록 가져오기
        all_files = glob.glob(os.path.join(class_path, '*', '*.png'))
        random.shuffle(all_files)

        total_files = len(all_files)
        if total_files == 0:
            print(f"경고: {class_path}에 PNG 파일이 없습니다.")
            continue

        # 분할 지점 계산
        train_end = int(total_files * split_ratios['train'])
        val_end = train_end + int(total_files * split_ratios['val'])

        # 파일 분할
        splits = {
            'train': all_files[:train_end],
            'val': all_files[train_end:val_end],
            'test': all_files[val_end:]
        }

        print(f"클래스 '{class_name}' (총 {total_files}개):")

        # 파일 복사
        for split_name, files in splits.items():
            dest_dir = os.path.join(output_dir, split_name, class_name)
            for f in files:
                shutil.copy(f, dest_dir)
            print(f"  - {split_name}: {len(files)}개 파일 복사 완료.")

    print("-" * 30)
    print("데이터 분할이 완료되었습니다.")

if __name__ == '__main__':
    split_data()
