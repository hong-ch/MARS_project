import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models.resnet import ResNet50_Weights
from torch.amp import autocast
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix, roc_curve, auc
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import re

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# 사용자 입력
batch_size = 4
num_classes = 3
test_dataset_root = 'C:/Users/smc/Desktop/CH/MARS_project/data/Recon_png_image_test'  # Test 데이터 폴더 경로
checkpoint_path = 'runs/ver2_Reconstructed/checkpoint/weights-best_model.pth'  # 학습된 가중치 경로

print(f"초기 GPU 메모리: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

# 커스텀 데이터셋
class CustomCTScanDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.patient_folders = []
        self.labels = []
        
        print(f"발견된 클래스: {self.classes}")
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for patient_folder in os.listdir(class_path):
                patient_path = os.path.join(class_path, patient_folder)
                if os.path.isdir(patient_path):
                    self.patient_folders.append(patient_path)
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, idx):
        patient_path = self.patient_folders[idx]
        label = self.labels[idx]
        
        slice_files = sorted([f for f in os.listdir(patient_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                             key=lambda x: int(re.match(r'(\d+)', x).group(1)) if re.match(r'(\d+)', x) else 0)
        
        num_slices = len(slice_files)
        if num_slices == 0:
            print(f"경고: 환자 폴더 {patient_path}에 유효한 이미지 없음")
            return None, None
        
        images = []
        if num_slices <= 30:
            selected_indices = range(num_slices)
        else:
            step = num_slices // 30
            selected_indices = [i * step for i in range(15)]
        
        for i in selected_indices:
            img_name = slice_files[i]
            img_path = os.path.join(patient_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"이미지 로드 실패: {img_path}")
                continue
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform is not None:  # transform이 None이 아니면 적용
                img = self.transform(img)
            images.append(img)
        
        if not images:
            print(f"경고: 환자 {patient_path}에서 유효한 이미지 로드 실패")
            return None, None
        
        images = torch.stack(images)  # Tensor로 변환된 images 스택
        return images, label

# 커스텀 collate_fn
def custom_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels)

# 변환
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Specificity 계산
def calculate_specificity(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    specificities = []
    for i in range(num_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
    return np.array(specificities)

# 모델 함수
def get_model(net, num_classes=3):
    if net == 'Resnet50':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # 기존 fc를 교체 (Dropout 추가)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),   # Dropout 추가
            nn.Linear(in_features, num_classes)
        )
        return model.to(device)

# Inference 함수
def inference(model, test_loader):
    model.eval()
    test_preds = []
    test_labels = []
    test_probs = []
    with torch.no_grad(), autocast('cuda'):
        test_bar = tqdm(test_loader, desc="Inference on Recon")
        for inputs, labels in test_bar:
            if inputs is None:
                continue
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size, num_selected, _, _, _ = inputs.shape
            outputs = torch.zeros(batch_size, num_classes).to(device)
            for i in range(num_selected):
                slice_inputs = inputs[:, i, :, :, :]
                slice_outputs = model(slice_inputs)
                outputs += slice_outputs / num_selected
            prob = F.softmax(outputs, dim=1).detach()
            pred = torch.argmax(prob, dim=1).cpu().numpy()
            test_preds.extend(pred)
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(prob.cpu().numpy())

    return np.array(test_preds), np.array(test_labels), np.array(test_probs)

if __name__ == '__main__':
    # 모델 로드
    model = get_model('Resnet50', num_classes)
    checkpoint_path = 'runs/ver2_Reconstructed/checkpoint/weights-best_model.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"가중치 로드 완료: {checkpoint_path}")
    else:
        print(f"가중치 파일을 찾을 수 없음: {checkpoint_path}")
        exit()

    # 테스트 데이터셋 로드
    test_dataset = CustomCTScanDataset(test_dataset_root, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    print(f"테스트 환자 수: {len(test_dataset)}")

    # Inference 수행
    test_preds, test_labels, test_probs = inference(model, test_loader)

    # 메트릭 계산
    test_acc = accuracy_score(test_labels, test_preds)
    try:
        test_roc_auc = roc_auc_score(test_labels, test_probs, multi_class='ovr')
    except ValueError:
        test_roc_auc = 0.0
    test_recall = recall_score(test_labels, test_preds, average=None, zero_division=0)
    test_specificity = calculate_specificity(test_labels, test_preds, num_classes)

    print(f'테스트 데이터 평가 결과:')
    print(f"ROC-AUC: {test_roc_auc:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Recall: {test_recall}")
    print(f"Specificity: {test_specificity}")

    # ROC-AUC 시각화
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--')  # 대각선 (무작위 분류 기준선)
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(test_labels, test_probs[:, i], pos_label=i)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve class {i} (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Reconstructed)')
    plt.legend(loc='best')
    plt.savefig('roc_curve_Recon.png')
    plt.close()

    # Confusion Matrix 시각화
    test_cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(test_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Reconstructed)')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, ['Normal', 'PN', 'TB'], rotation=45)
    plt.yticks(tick_marks, ['Normal', 'PN', 'TB'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    thresh = test_cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(test_cm[i, j]), ha='center', va='center', color='white' if test_cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.savefig('confusion_matrix_Recon.png')
    plt.close()

    print(f"Test Confusion Matrix:\n{test_cm}")