import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet34_Weights
from sklearn.metrics import precision_score, confusion_matrix, f1_score, recall_score
import numpy as np

"""
This script is a template for training a deep learning model on a dataset of harmonic signal images,
derived from IQ samples.

We use ResNet34 as the base model

This version is designed to run on a single machine, with or without GPU support.
"""

def main():
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transformations for training and validation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dynamic paths based on the current working directory
    current_dir = os.getcwd()
    train_dir = os.path.join(current_dir, "images/train")
    val_dir = os.path.join(current_dir, "images/val")
    test_dir = os.path.join(current_dir, "images/test")
    best_model_path = os.path.join(current_dir, "best_model.pth")
    saved_model_path = os.path.join(current_dir, "resnet34_usbfingerprinter.pth")

    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Load pre-trained ResNet
    model = models.resnet34(weights=ResNet34_Weights.DEFAULT)

    # Modify the final layer to match the number of classes in the dataset
    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

            # Evaluate on the validation set
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

            # Save the model if it improves
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(model.state_dict(), best_model_path)

        print(f"Best Validation Accuracy: {best_accuracy:.4f}")

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=32)

    # Evaluate the model
    def evaluate_model(model, test_loader):
        model.eval()
        test_corrects = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                test_corrects += torch.sum(preds == labels.data)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = test_corrects.double() / len(test_loader.dataset)
        print(f"Test Accuracy: {test_acc:.4f}")

        # Calculate precision
        precision = precision_score(all_labels, all_preds, average='weighted')
        print(f"Precision: {precision:.4f}")
        
        # Calculate recall
        recall = recall_score(all_labels, all_preds, average='weighted')
        print(f"Recall: {recall:.4f}")

        # Calculate F1-score
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"F1-Score: {f1:.4f}")

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:")
        print(conf_matrix)

    # Load the best model
    model.load_state_dict(torch.load(best_model_path))

    # Evaluate the model
    evaluate_model(model, test_loader)

    # Save the entire model
    torch.save(model, saved_model_path)

if __name__ == "__main__":
    main()
