# evaluation.py

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from models.resnet50 import load_resnet50
import os

def evaluate_model(test_folder, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ImageFolder(root=test_folder, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_resnet50()
    model.load_state_dict(torch.load("resnet50.pth"))  # Load trained model
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


evaluate_model("data/Celeb-real_frames")
evaluate_model("data/Celeb-synthesis_frames")
