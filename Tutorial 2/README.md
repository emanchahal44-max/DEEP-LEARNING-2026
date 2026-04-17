# Multi-Layer Perceptron (MLP) Classification on Iris Dataset

This project demonstrates the implementation and analysis of a Multi-Layer Perceptron (MLP) classifier using the Iris dataset.

---

## Project Objectives

- Train an MLP classifier on the Iris dataset  
- Apply feature scaling using StandardScaler  
- Compare different hidden layer architectures  
- Analyze the effect of learning rate on convergence  
- Visualize learning curves  
- Evaluate model performance using classification metrics  

---

## Dataset Information

The Iris dataset contains:

- 150 samples  
- 4 input features  
- 3 classes:
  - Setosa  
  - Versicolor  
  - Virginica  

The dataset was split into:
- 80% training data  
- 20% testing data  

All features were standardized before training.

---

## Model Architectures Tested

Three different hidden layer configurations were evaluated:

- (5, 5)  
- (10, 10)  
- (15, 15)  

### Accuracy Results

| Architecture | Accuracy |
|-------------|----------|
| (5, 5)      | 1.00 |
| (10, 10)    | 0.97 |
| (15, 15)    | 1.00 |

### Observations

- Increasing the number of neurons did not significantly improve accuracy.
- Smaller architectures were sufficient for perfect classification.
- The dataset is simple and well-separated, so deep networks are unnecessary.

---

## Learning Rate Experiment

A high learning rate (0.1) was tested using the (10,10) architecture.

### Results

- Accuracy: 1.00  
- Precision: 1.00  
- Recall: 1.00  
- F1-score: 1.00  

Although final accuracy was perfect, the learning curve showed fluctuations due to large weight updates.

### Learning Rate Insights

- Small learning rate → Slow but stable convergence  
- Moderate learning rate (0.001–0.01) → Balanced and smooth training  
- High learning rate (0.1) → Faster convergence but unstable loss curve  

---

## Visualizations

The project includes:

- Learning curves for different architectures  
- Learning curve for high learning rate model  
- Saved graph images for comparison  

---

## 🛠️ Technologies Used

- Python  
- Scikit-learn  
- Matplotlib  

---

## Conclusions

- Moderate architectures such as (5,5) or (10,10) provide optimal performance.  
- Increasing network size does not guarantee better accuracy for simple datasets.  
- Learning rate significantly affects convergence behavior.  
- A balanced learning rate provides the best stability and efficiency.  

