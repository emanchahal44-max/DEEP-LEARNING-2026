
# MNIST Digit Classification with PyTorch

This repository contains a fully connected neural network implementation in **PyTorch** for **classifying handwritten digits** from the MNIST dataset. The project demonstrates **key neural network concepts**, including network architecture design, activation functions, regularization techniques, and performance visualization.

---

## Project Overview

- **Dataset:** MNIST (28x28 grayscale images of digits 0–9)  
- **Objective:** Accurately classify handwritten digits using a neural network  
- **Approach:**  
  1. Load and preprocess MNIST dataset  
  2. Define a fully connected network (`MNISTNet`) with **two hidden layers**  
  3. Train the model using cross-entropy loss and Adam optimizer  
  4. Evaluate on validation and test sets  
  5. Visualize training/validation loss and accuracy  
  6. Make predictions on test images and visualize results  

---

## ⚙️ Model Architecture

| Layer | Output Size | Activation | Notes |
|-------|------------|-----------|------|
| Input | 784 | - | Flattened 28x28 image |
| FC1   | 128 | ReLU | Dropout 0.2 applied |
| FC2   | 64  | ReLU | Dropout 0.2 applied |
| Output | 10  | - | Raw logits for 10 classes |

---

## Parameters and Design Choices

- **Hidden Layers:** Two layers (128 → 64 neurons) balance **complexity** and **performance**.  
- **Activation Function:** ReLU introduces non-linearity and prevents vanishing gradient issues.  
- **Dropout (0.2):** Randomly disables 20% of neurons during training to reduce overfitting by forcing the model to rely on multiple pathways rather than memorizing patterns.  
- **Weight Decay (1e-4):** L2 regularization penalizes large weights, further improving generalization.  
- **Optimizer:** Adam with learning rate 0.001 for adaptive and efficient updates.  
- **Loss Function:** Cross-Entropy Loss for multi-class classification.  
- **Batch Size:** 32  
- **Epochs:** 10 (sufficient to achieve high accuracy without overfitting)  

**Why Dropout + Weight Decay Together?**  
Combining these two regularization techniques ensures the model is **well-fitted**:  
- Dropout prevents the network from memorizing the training set (avoiding overfitting).  
- Weight decay prevents weights from becoming too large, smoothing the learned function.  
As a result, **training and validation curves stay close together**, and the model generalizes well to unseen data.

---

## Training and Validation

- The model was trained on **80% of the MNIST training set**, with 20% reserved for validation.  
- **Observations:**  
  - Training and validation loss decrease smoothly and remain **very close**, indicating **no overfitting**.  
  - Training and validation accuracy rise together (~97%), indicating **no underfitting**.  
  - Dropout and weight decay help maintain this balance.  


## Results and Analysis

After training the neural network on the MNIST dataset, the following observations were made:

- **Training and Validation Performance:**  
  - Training and validation **loss decreased smoothly** over epochs.  
  - Training and validation **accuracy increased together**, staying very close (~97%).  
  - This indicates the model is **well-fitted**, showing **no overfitting** (validation loss is not rising) and **no underfitting** (accuracy is high for both sets).  





