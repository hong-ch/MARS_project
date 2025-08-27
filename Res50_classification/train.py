
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from torchvision.models.resnet import ResNet50_Weights
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.amp import GradScaler, autocast
import os
import cv2
import numpy as np
import pandas as pd
import math
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
import time
import torch.nn.functional as F
import re
from tqdm import tqdm

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# 사용자 입력
Number_of_folds = 4
batch_size = 4
epochs = 50
Disease_name = 'Resnet50_Edena_Binary_CL200withAgumentation2'
num_classes = 3
dataset_root = 'C:/Users/smc/Desktop/CH/MARS_project/data/Recon_png_image'

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
            if self.transform:
                img = self.transform(img)
            images.append(img)
        
        if not images:
            print(f"경고: 환자 {patient_path}에서 유효한 이미지 로드 실패")
            return None, None
        
        images = torch.stack(images)
        return images, label

# 커스텀 collate_fn
def custom_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels)

# 변환
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

# 학습 함수
def train_model(model, train_loader, val_loader, epochs, fold):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    scaler = GradScaler('cuda')

    # TensorBoard 로그 디렉토리 변경 (version 기반)
    log_dir = f"runs/ver8_15slices_pickall_Fold-{fold}_lr=0.1^4_drop(0.5)-Recon"
    writer = SummaryWriter(log_dir=log_dir)
    
    # Best model 저장 경로 변경 (checkpoint 하위 폴더)
    checkpoint_dir = os.path.join(log_dir, 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f'weights-best_model-Fold-{fold}.pth')
    
    best_roc_auc = 0.0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_roc_auc': [], 'val_roc_auc': [],
        'train_recall': [], 'val_recall': [],
        'train_specificity': [], 'val_specificity': []
    }
    patience = 300
    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        train_probs = []
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch")
        for inputs, labels in train_bar:
            if inputs is None:
                continue
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            batch_size, num_selected, _, _, _ = inputs.shape
            outputs = torch.zeros(batch_size, num_classes).to(device)
            with autocast('cuda'):
                for i in range(num_selected):
                    slice_inputs = inputs[:, i, :, :, :]
                    slice_outputs = model(slice_inputs)
                    outputs += slice_outputs / num_selected
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * batch_size
            prob = F.softmax(outputs, dim=1).detach()
            _, predicted = outputs.max(1)
            total += batch_size
            correct += predicted.eq(labels).sum().item()
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_probs.extend(prob.cpu().numpy())
            train_bar.set_postfix(loss=loss.item(), acc=correct/total)

        train_loss = running_loss / total if total > 0 else float('inf')
        train_acc = correct / total if total > 0 else 0.0
        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
        train_probs = np.array(train_probs)
        
        # 학습 메트릭 계산
        try:
            train_roc_auc = roc_auc_score(train_labels, train_probs, multi_class='ovr')
        except ValueError:
            train_roc_auc = 0.0
        train_recall = recall_score(train_labels, train_preds, average=None, zero_division=0)
        train_specificity = calculate_specificity(train_labels, train_preds, num_classes)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_roc_auc'].append(train_roc_auc)
        history['train_recall'].append(train_recall)
        history['train_specificity'].append(train_specificity)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_preds = []
        val_labels = []
        val_probs = []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", unit="batch")
        with torch.no_grad():
            for inputs, labels in val_bar:
                if inputs is None:
                    continue
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size, num_selected, _, _, _ = inputs.shape
                outputs = torch.zeros(batch_size, num_classes).to(device)
                with autocast('cuda'):
                    for i in range(num_selected):
                        slice_inputs = inputs[:, i, :, :, :]
                        slice_outputs = model(slice_inputs)
                        outputs += slice_outputs / num_selected
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * batch_size
                prob = F.softmax(outputs, dim=1).detach()
                _, predicted = outputs.max(1)
                total += batch_size
                correct += predicted.eq(labels).sum().item()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(prob.cpu().numpy())
                val_bar.set_postfix(loss=loss.item(), acc=correct/total)

        val_loss = val_loss / total if total > 0 else float('inf')
        val_acc = correct / total if total > 0 else 0.0
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_probs = np.array(val_probs)
        
        # 검증 메트릭 계산
        try:
            val_roc_auc = roc_auc_score(val_labels, val_probs, multi_class='ovr')
        except ValueError:
            val_roc_auc = 0.0
        val_recall = recall_score(val_labels, val_preds, average=None, zero_division=0)
        val_specificity = calculate_specificity(val_labels, val_preds, num_classes)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_roc_auc'].append(val_roc_auc)
        history['val_recall'].append(val_recall)
        history['val_specificity'].append(val_specificity)

        # TensorBoard 로깅
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('ROC_AUC/train', train_roc_auc, epoch)
        writer.add_scalar('ROC_AUC/val', val_roc_auc, epoch)
        for i in range(num_classes):
            writer.add_scalar(f'Recall/train_class_{i}', train_recall[i], epoch)
            writer.add_scalar(f'Recall/val_class_{i}', val_recall[i], epoch)
            writer.add_scalar(f'Specificity/train_class_{i}', train_specificity[i], epoch)
            writer.add_scalar(f'Specificity/val_class_{i}', val_specificity[i], epoch)

        # Best model 저장 (val_roc_auc 기준)
        if val_roc_auc > best_roc_auc:
            best_roc_auc = val_roc_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'에포크 {epoch+1}에서 조기 종료')
                break

        scheduler.step()

        # 에포크 결과 출력
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} 완료, 시간: {epoch_time:.2f}초")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train ROC-AUC: {train_roc_auc:.4f}")
        print(f"Train Recall: {train_recall}, Train Specificity: {train_specificity}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val ROC-AUC: {val_roc_auc:.4f}")
        print(f"Val Recall: {val_recall}, Val Specificity: {val_specificity}")

    writer.close()
    pd.DataFrame(history).to_csv(f'{Disease_name}-History-Fold-{fold}.csv', index=False)
    return model

if __name__ == '__main__':
    full_dataset = CustomCTScanDataset(dataset_root, transform=None)
    print(f"총 환자 수: {len(full_dataset)}")
    print(f"클래스 매핑: {full_dataset.class_to_idx}")
    
    # 첫 5개 환자 폴더의 슬라이스 수 확인
    for patient_path in full_dataset.patient_folders[:5]:
        print(f"{patient_path}: {len(os.listdir(patient_path))} 슬라이스")
    
    kf = KFold(n_splits=Number_of_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_dataset))), 1):
        print(f'----------- 폴드-{fold} 데이터 로드 시작 ----------------')

        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, 
                                 collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, 
                                collate_fn=custom_collate_fn)

        print(f'학습 환자: {len(train_idx)}, 검증 환자: {len(val_idx)}')
        print(f'----------- 폴드-{fold} 데이터 로드 완료 ----------------')

        model = get_model('Resnet50', num_classes)

        print('----학습 시작-------')
        trained_model = train_model(model, train_loader, val_loader, epochs, fold)

        trained_model.eval()
        preds = []
        probs = []
        true_labels = []
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="예측", unit="batch")
            for inputs, labels in val_bar:
                if inputs is None:
                    continue
                inputs = inputs.to(device)
                batch_size, num_selected, _, _, _ = inputs.shape
                outputs = torch.zeros(batch_size, num_classes).to(device)
                for i in range(num_selected):
                    slice_inputs = inputs[:, i, :, :, :]
                    slice_outputs = trained_model(slice_inputs)
                    outputs += slice_outputs / num_selected
                prob = F.softmax(outputs, dim=1).detach()
                pred = np.argmax(prob.cpu().numpy(), axis=1)
                preds.extend(pred)
                probs.extend(prob.cpu().numpy())
                true_labels.extend(labels.numpy())

        preds = np.array(preds)
        probs = np.array(probs)
        true_labels = np.array(true_labels)

        acc = accuracy_score(true_labels, preds)
        try:
            final_roc_auc = roc_auc_score(true_labels, probs, multi_class='ovr')
        except ValueError:
            final_roc_auc = 0.0
        final_recall = recall_score(true_labels, preds, average=None, zero_division=0)
        final_specificity = calculate_specificity(true_labels, preds, num_classes)

        print(f'환자 수준 테스트 정확도 (평균 확률): {acc:.4f}')
        print(f'최종 ROC-AUC: {final_roc_auc:.4f}')
        print(f'최종 Recall: {final_recall}')
        print(f'최종 Specificity: {final_specificity}')

        roc_name = f'{Disease_name}_roc-Fold-{fold}.csv'
        roc_df = pd.DataFrame({
            'pred': list(probs),
            'pred0': probs[:, 0],
            'pred1': probs[:, 1],
            'pred2': probs[:, 2],
            'prediction': preds,
            'y_true': true_labels
        })
        roc_df.to_csv(roc_name, index=False)

        pred_true_name = f'{Disease_name}_PredictionAndTruelabel-Fold-{fold}.csv'
        pred_true_df = pd.DataFrame({'pre': preds, 'lbls': true_labels})
        pred_true_df.to_csv(pred_true_name, index=False)

        print("모델과 가중치가 저장되었습니다\n")
        print(f'#----------- 폴드-{fold} 데이터 학습 완료 ----------------#\n')

    print('----------------------학습 종료------------------------------')