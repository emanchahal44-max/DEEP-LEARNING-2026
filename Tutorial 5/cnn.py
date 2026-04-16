import sys
print("STARTING CNN TRAINING...")
sys.stdout.flush()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ===============================
# Device
# ===============================
device = torch.device("cpu")
print("Using device:", device)

# ===============================
# Data Augmentation
# ===============================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# ===============================
# Dataset
# ===============================
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform_test
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ===============================
# Efficient CNN Architecture
# ===============================
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        # Convolution Blocks
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2,2)

        # Global Average Pooling (VERY IMPORTANT)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(128, 10)

    def forward(self,x):

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.gap(x)          # shape -> (batch, 128,1,1)
        x = torch.flatten(x,1)   # shape -> (batch,128)

        x = self.dropout(x)
        x = self.fc(x)

        return x

# ===============================
# Initialize Model
# ===============================
model = CNN().to(device)
print(model)

# ===============================
# Loss + Optimizer
# ===============================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=5e-4
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)

# ===============================
# Training Setup
# ===============================
epochs = 10

train_losses=[]
test_losses=[]
train_accs=[]
test_accs=[]

print("Starting Training Process...")

# ===============================
# Training Loop
# ===============================
for epoch in range(epochs):

    model.train()
    running_loss=0
    correct=0
    total=0

    for images,labels in train_loader:

        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()

        outputs=model(images)
        loss=criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        running_loss+=loss.item()

        _,pred=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(pred==labels).sum().item()

    train_loss=running_loss/len(train_loader)
    train_acc=100*correct/total

    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Evaluation
    model.eval()
    correct=0
    total=0
    val_loss=0

    with torch.no_grad():
        for images,labels in test_loader:

            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images)
            loss=criterion(outputs,labels)

            val_loss+=loss.item()

            _,pred=torch.max(outputs,1)
            total+=labels.size(0)
            correct+=(pred==labels).sum().item()

    test_loss=val_loss/len(test_loader)
    test_acc=100*correct/total

    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"Epoch [{epoch+1}/{epochs}] | "
          f"Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Test Acc: {test_acc:.2f}%")

    scheduler.step()

print("Training finished")
print(f"Final Test Accuracy: {test_acc:.2f}%")

# ===============================
# Plot Curves
# ===============================
plt.figure()
plt.plot(train_losses,label="Train Loss")
plt.plot(test_losses,label="Test Loss")
plt.legend()
plt.savefig("loss_curve.png")
plt.close()

plt.figure()
plt.plot(train_accs,label="Train Accuracy")
plt.plot(test_accs,label="Test Accuracy")
plt.legend()
plt.savefig("accuracy_curve.png")
plt.close()

print("Plots saved.")