import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import models, datasets, transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Dataset Paths
train_path = "dataset/train"
test_path = "dataset/test"

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# Dataset Loader
train_dataset = datasets.ImageFolder(train_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Model
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Freeze layers
for param in model.parameters():
    param.requires_grad = False

# Modify classifier
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training tracking
epochs = 5
loss_list = []
acc_list = []

# Training loop
for epoch in range(epochs):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    loss_list.append(total_loss/len(train_loader))
    acc_list.append(correct/total)

    print(f"Epoch {epoch+1} | Loss: {loss_list[-1]:.4f} | Accuracy: {acc_list[-1]:.4f}")

# Plot Accuracy Curve
plt.figure(figsize=(6,4))
plt.plot(range(1,epochs+1), acc_list, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Curve (Transfer Learning - ResNet50)")
plt.grid(True)
plt.tight_layout()

plt.savefig("accuracy_curve.png")   # saves graph
plt.show()


# Plot Loss Curve
plt.figure(figsize=(6,4))
plt.plot(range(1,epochs+1), loss_list, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve (Transfer Learning - ResNet50)")
plt.grid(True)
plt.tight_layout()

plt.savefig("loss_curve.png")   # saves graph
plt.show()