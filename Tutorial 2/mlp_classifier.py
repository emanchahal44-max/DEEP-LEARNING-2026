from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# Architecture Comparison
# ==============================

neurons_list = [(5,5), (10,10), (15,15)]

plt.figure(figsize=(8,6))

for neurons in neurons_list:
    mlp = MLPClassifier(hidden_layer_sizes=neurons,
                        max_iter=1000,
                        learning_rate_init=0.001,
                        random_state=42)

    mlp.fit(X_train_scaled, y_train)
    y_pred = mlp.predict(X_test_scaled)

    print(f"Neurons {neurons} -> Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    plt.plot(mlp.loss_curve_, label=f'Layers {neurons}')

plt.title("Learning Curves for Different Architectures")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("architecture_comparison.png", dpi=300)
plt.show()

mlp_high_lr = MLPClassifier(hidden_layer_sizes=(10,10),
                            max_iter=1000,
                            learning_rate_init=0.1,
                            random_state=42)

mlp_high_lr.fit(X_train_scaled, y_train)
y_pred_high = mlp_high_lr.predict(X_test_scaled)

print("\nHigh Learning Rate Accuracy:",
      accuracy_score(y_test, y_pred_high))

plt.figure(figsize=(8,6))
plt.plot(mlp_high_lr.loss_curve_, label="High LR (0.1)")
plt.title("Learning Curve with High Learning Rate")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("high_learning_rate.png", dpi=300)
plt.show()
print("\nClassification Report (High LR Model):")
print(classification_report(y_test, y_pred_high))