# Perceptron Classifier from Scratch (Iris Dataset) 

This repository contains an implementation of a **Perceptron classifier from scratch** using **NumPy**, without using built-in machine learning models.  
The model is trained and evaluated on the **Iris dataset** for **binary classification**.

## Project Description

This project demonstrates the working of a perceptron by implementing:
- Weight initialization
- Weighted sum calculation
- Sigmoid activation function
- Gradient-based learning
- Binary classification decision rule

The classifier predicts whether a flower belongs to the **Iris-setosa** class or not.

## Key Features

- Perceptron implemented completely from scratch
- Sigmoid activation function
- Binary classification (Setosa vs Non-Setosa)
- Manual user input for prediction
- Model evaluation using accuracy
- Uses a real-world dataset (Iris)

## File Structure

├── Perceptron.py
├── README.md

## Dataset Information

- **Dataset**: Iris Dataset  
- **Source**: UCI Machine Learning Repository  
- **Total Samples**: 150  
- **Features**:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- **Classes**:
  - Iris-setosa
  - Iris-versicolor
  - Iris-virginica

Label encoding used:
- Iris-setosa → `1`
- Other species → `0`

---

## Model Details

- Learning rate (η): 0.01  
- Number of iterations: 50  
- Activation function: Sigmoid  
- Classification threshold: 0.5  

Weights are updated using the difference between predicted output and actual target.


## How to Run

1. Clone the repository or download the files
2. Open the project directory
3. Run the program:

```bash
python Perceptron.py
```

4. Enter the flower measurements when prompted:
   - Sepal length
   - Sepal width
   - Petal length
   - Petal width

---

## Output

The program displays:
- Predicted flower class
- Test accuracy of the model

Example output:

```
Enter flower measurements:
Sepal length: 5.1
Sepal width: 3.5
Petal length: 1.4
Petal width: 0.2

Prediction: Iris-setosa

Test Accuracy: 100 %
```

## Learning Outcomes

- Understanding perceptron learning from scratch
- Binary classification concepts
- Effect of data separability on accuracy
- Importance of model evaluation
- Difference between simple and complex ML problems




