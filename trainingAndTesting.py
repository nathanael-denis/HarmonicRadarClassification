import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import os
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
import random
import time

# -------------------- Parameters --------------------
LIMIT_FRACTION = 1  # fraction of images to use, 1 is the whole dataset
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 45
BASE_DIR = "output"  # adjust path if needed

# -------------------- Unique checkpoint filename --------------------
RUN_ID = f"{os.getpid()}_{int(time.time())}"
BEST_MODEL_PATH = f"best_model_{RUN_ID}.pth"

# -------------------- Transformations --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # MobileNetV2 expects 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------- Load datasets --------------------
train_dataset = datasets.ImageFolder(os.path.join(BASE_DIR, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(BASE_DIR, "val"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(BASE_DIR, "test"), transform=transform)
ood_dataset = datasets.ImageFolder(os.path.join(BASE_DIR, "ood"), transform=transform)

# -------------------- Limit dataset size --------------------
def limit_dataset(dataset, fraction=0.4, seed=42):
    num_samples = len(dataset)
    num_keep = int(num_samples * fraction)
    random.seed(seed)
    indices = random.sample(range(num_samples), num_keep)
    return Subset(dataset, indices)

train_dataset = limit_dataset(train_dataset, fraction=LIMIT_FRACTION, seed=SEED)
val_dataset = limit_dataset(val_dataset, fraction=LIMIT_FRACTION, seed=SEED)
test_dataset = limit_dataset(test_dataset, fraction=LIMIT_FRACTION, seed=SEED)
ood_dataset = limit_dataset(ood_dataset, fraction=LIMIT_FRACTION, seed=SEED)

# -------------------- Data loaders --------------------
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
ood_loader = DataLoader(ood_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------- Model --------------------
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
num_classes = len(train_dataset.dataset.classes)  # Subset wraps the original dataset

# Replace classifier
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------------------- Training loop --------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=32):
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        # Validation
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Validation Acc: {val_acc:.4f}")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")

# -------------------- Evaluation --------------------
def evaluate_model(model, data_loader, class_names, name="Test"):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"\n{name} Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    print("Confusion Matrix:")
    print(cm)

# -------------------- Main --------------------
if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS)

    # Load best model (from this run only)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))

    # Evaluate on test set
    evaluate_model(model, test_loader, test_dataset.dataset.classes, name="Test")

    # Evaluate on OOD set
    evaluate_model(model, ood_loader, ood_dataset.dataset.classes, name="OOD")

