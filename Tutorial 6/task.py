import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import (
    VGG16_Weights,
    ResNet50_Weights,
    AlexNet_Weights,
    ResNet101_Weights,
    MobileNet_V2_Weights
)
from PIL import Image

# --------------------------------------------------
# STEP 1: Load ImageNet Class Labels
# --------------------------------------------------
with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# --------------------------------------------------
# STEP 2: Preprocess Image Function
# --------------------------------------------------
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# --------------------------------------------------
# STEP 3: Load Pretrained Models
# --------------------------------------------------

vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)

alexnet = models.alexnet(weights=AlexNet_Weights.DEFAULT)
resnet101 = models.resnet101(weights=ResNet101_Weights.DEFAULT)
mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

# Set evaluation mode
vgg16.eval()
resnet50.eval()
alexnet.eval()
resnet101.eval()
mobilenet.eval()

# --------------------------------------------------
# STEP 4: Prediction Function
# --------------------------------------------------
def predict_top5(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)

    return top5_prob, top5_catid

# --------------------------------------------------
# STEP 5: Image Path (CHANGE THIS IF NEEDED)
# --------------------------------------------------
image_path = r"C:\Users\SAAD COMPUTER\OneDrive\Desktop\Task 6\LION2.webp"

# Preprocess image
image_tensor = preprocess_image(image_path)

# --------------------------------------------------
# STEP 6: Get Predictions
# --------------------------------------------------

vgg_probs, vgg_ids = predict_top5(vgg16, image_tensor)
res_probs, res_ids = predict_top5(resnet50, image_tensor)

alex_probs, alex_ids = predict_top5(alexnet, image_tensor)
res101_probs, res101_ids = predict_top5(resnet101, image_tensor)
mob_probs, mob_ids = predict_top5(mobilenet, image_tensor)

# --------------------------------------------------
# STEP 7: Print Results
# --------------------------------------------------

print("\nTop 5 predictions from VGG16:")
for i in range(5):
    label = classes[vgg_ids[i]]
    prob = vgg_probs[i].item() * 100
    print(f"{i+1}. {label} ({prob:.2f}% probability)")

print("\nTop 5 predictions from ResNet50:")
for i in range(5):
    label = classes[res_ids[i]]
    prob = res_probs[i].item() * 100
    print(f"{i+1}. {label} ({prob:.2f}% probability)")

print("\nTop 5 predictions from AlexNet:")
for i in range(5):
    label = classes[alex_ids[i]]
    prob = alex_probs[i].item() * 100
    print(f"{i+1}. {label} ({prob:.2f}% probability)")

print("\nTop 5 predictions from ResNet101:")
for i in range(5):
    label = classes[res101_ids[i]]
    prob = res101_probs[i].item() * 100
    print(f"{i+1}. {label} ({prob:.2f}% probability)")

print("\nTop 5 predictions from MobileNetV2:")
for i in range(5):
    label = classes[mob_ids[i]]
    prob = mob_probs[i].item() * 100
    print(f"{i+1}. {label} ({prob:.2f}% probability)")