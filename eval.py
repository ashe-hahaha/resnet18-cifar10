import torch
import torch.nn as nn
from torchvision import models
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, f1_score, accuracy_score
import os
import torchvision.datasets as datasets

MODEL_PATH = "saved/resnet18_cifar10_v3.pth"

# load and modify model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)  # modify fc layer
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# load testset
if not os.path.exists("saved/cifar10_testset.pth"):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    torch.save(testset, "saved/cifar10_testset.pth")


testset = torch.load("saved/cifar10_testset.pth", weights_only=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

device = torch.device("cuda")
model.to(device)

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())


accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")
print(f"Total Accuracy: {accuracy:.4f}")
print(f"Total F1 Score: {f1:.4f}")