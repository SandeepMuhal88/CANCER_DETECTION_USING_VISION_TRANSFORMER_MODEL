import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DIR   = r"../Data/chest_xray/chest_xray/train"
VAL_DIR     = r"../Data/chest_xray/chest_xray/val"
MODEL_OUT   = r"../models/xray_model.pth"
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 1e-4
NUM_CLASSES = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
print("[OK] Using device:", device)

# ── Transforms ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ── Datasets & Loaders ────────────────────────────────────────────────────────
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                           shuffle=True, num_workers=0)

has_val = os.path.isdir(VAL_DIR) and len(os.listdir(VAL_DIR)) > 0
if has_val:
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

print("[DATA] Train:", len(train_dataset), "images | Classes:", train_dataset.classes)
if has_val:
    print("[DATA] Val  :", len(val_dataset), "images")

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# ── Loss & Optimiser ──────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ── Training Loop ─────────────────────────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct    = 0

    for images, labels in tqdm(train_loader,
                               desc=f"Epoch {epoch+1}/{EPOCHS} [Train]",
                               leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()

    train_loss = total_loss / len(train_dataset)
    train_acc  = correct   / len(train_dataset) * 100

    if has_val:
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs      = model(images)
                val_loss    += criterion(outputs, labels).item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss = val_loss   / len(val_dataset)
        val_acc  = val_correct / len(val_dataset) * 100

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"   [SAVED] Best model  (val acc {val_acc:.1f}%)")
    else:
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1f}%")

    scheduler.step()

if not has_val:
    torch.save(model.state_dict(), MODEL_OUT)
    print("[SAVED] Model saved ->", MODEL_OUT)

print("[DONE] Training complete!")