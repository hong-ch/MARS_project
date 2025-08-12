import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import sys

# 환자 단위 Dataset (슬라이스 목록 그룹화)
class PatientLevelDataset(Dataset):
    def __init__(self, root_dir, transform=None, patient_list=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = {'Normal': 0, 'PN': 1, 'TB': 2}
        self.patients = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                print(f"Directory not found: {cls_dir}", file=sys.stdout, flush=True)
                continue
            for patient in os.listdir(cls_dir):
                if patient_list is None or patient in patient_list:
                    patient_dir = os.path.join(cls_dir, patient)
                    slices = sorted(glob.glob(os.path.join(patient_dir, "*.png")))
                    self.patients.append((patient, self.classes[cls], slices))

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id, label, slices = self.patients[idx]
        slice_tensors = []
        for slice_path in slices:
            img = Image.open(slice_path).convert('L')
            if self.transform:
                img = self.transform(img)
            slice_tensors.append(img)
        return patient_id, torch.stack(slice_tensors), label

# Train/Test 분할 함수
def split_dataset(root_dir):
    all_patients = []
    for cls in ['Normal', 'PN', 'TB']:
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"Directory not found: {cls_dir}", file=sys.stdout, flush=True)
            continue
        patients = [p for p in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, p))]
        all_patients.extend([(p, cls) for p in patients])

    if not all_patients:
        raise ValueError("No patients found in the dataset.")
    train_patients, test_patients = train_test_split(
        all_patients, test_size=0.2, stratify=[p[1] for p in all_patients], random_state=42
    )
    train_patients = [p[0] for p in train_patients]
    test_patients = [p[0] for p in test_patients]
    return train_patients, test_patients

# 모델: 2D ResNet-18 (pretrained)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

# 훈련/테스트 함수 (환자 수준 예측)
def predict_patient(model, slices_tensor, device='cuda'):
    model.eval()
    model.to(device)
    probs = []
    with torch.no_grad():
        for slice_tensor in slices_tensor:
            slice_tensor = slice_tensor.unsqueeze(0).to(device)
            output = model(slice_tensor)
            prob = torch.softmax(output, dim=1).cpu().numpy()
            probs.append(prob)
    avg_prob = np.mean(probs, axis=0)
    return np.argmax(avg_prob)

# 훈련 함수 (진행 상황 추가)
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for i, (patient_id, slices, label) in enumerate(train_loader):
        print(f"Processing patient {i+1}/{len(train_loader)}", file=sys.stdout, flush=True)
        slices, label = slices.to(device), label.to(device)
        optimizer.zero_grad()
        loss = 0
        for slice_tensor in slices.squeeze(0):
            output = model(slice_tensor.unsqueeze(0))
            loss += criterion(output, label)
        avg_loss = loss / slices.size(0)
        avg_loss.backward()
        optimizer.step()
        total_loss += avg_loss.item()
        print(f"Memory allocated after batch: {torch.cuda.memory_allocated() / 1024**2:.2f} MB", file=sys.stdout, flush=True)
    return total_loss / len(train_loader)

# 평가 함수 (진행 상황 추가)
def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, (patient_id, slices, label) in enumerate(data_loader):
            print(f"Evaluating patient {i+1}/{len(data_loader)}", file=sys.stdout, flush=True)
            slices = slices.squeeze(0).to(device)
            pred = predict_patient(model, slices, device)
            all_preds.append(pred)
            all_labels.append(label.item())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    specificity = []
    for i in range(len(cm)):
        tn = np.sum(cm) - np.sum(cm[i]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    avg_specificity = np.mean(specificity)
    return accuracy, f1, recall, avg_specificity

# 사용 예시
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

root_dir = r"C:\Users\user1\Desktop\Changhee\project\data\LDCT_png_test"
train_patients, test_patients = split_dataset(root_dir)

train_dataset = PatientLevelDataset(root_dir=root_dir, transform=transform, patient_list=train_patients)
print(f"Number of train patients: {len(train_dataset)}", file=sys.stdout, flush=True)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

test_dataset = PatientLevelDataset(root_dir=root_dir, transform=transform, patient_list=test_patients)
print(f"Number of test patients: {len(test_dataset)}", file=sys.stdout, flush=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", file=sys.stdout, flush=True)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Memory allocated before training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB", file=sys.stdout, flush=True)

num_epochs = 2
for epoch in range(num_epochs):
    print(f"Starting Epoch {epoch+1}/{num_epochs}", file=sys.stdout, flush=True)
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Training completed for epoch {epoch+1}", file=sys.stdout, flush=True)
    print(f"Memory allocated after training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB", file=sys.stdout, flush=True)
    train_acc, train_f1, train_recall, train_spec = evaluate(model, train_loader, device)
    test_acc, test_f1, test_recall, test_spec = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1}/{num_epochs}:", file=sys.stdout, flush=True)
    print(f"Train Loss: {train_loss:.4f}", file=sys.stdout, flush=True)
    print(f"Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Train Recall: {train_recall:.4f}, Train Specificity: {train_spec:.4f}", file=sys.stdout, flush=True)
    print(f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test Recall: {test_recall:.4f}, Test Specificity: {test_spec:.4f}", file=sys.stdout, flush=True)

print("Final evaluation on test set:", file=sys.stdout, flush=True)
for patient_id, slices, label in test_loader:
    slices = slices.squeeze(0)
    pred = predict_patient(model, slices, device)
    print(f"환자 {patient_id}: 예측 클래스 {pred}, 실제 {label.item()}", file=sys.stdout, flush=True)