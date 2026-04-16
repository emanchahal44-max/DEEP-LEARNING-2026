import numpy as np

class PerceptronScratch:
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def weighted_sum(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    #Sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        z = self.weighted_sum(X)
        return 1 if self.sigmoid(z) >= 0.5 else 0

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                z = self.weighted_sum(xi)
                y_pred = self.sigmoid(z)
                
                update = self.eta * (target - y_pred)
                self.w_[1:] += update * xi
                self.w_[0] += update

                errors += int(abs(target - y_pred) > 0.5)
            self.errors_.append(errors)
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)

df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

df = shuffle(df, random_state=42)

X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

train_data, test_data, train_labels, test_labels = train_test_split(
    X, y, test_size=0.25, random_state=42
)

train_labels = np.where(train_labels == 'Iris-setosa', 1, 0)
test_labels = np.where(test_labels == 'Iris-setosa', 1, 0)
ppn = PerceptronScratch(eta=0.01, n_iter=50)
ppn.fit(train_data, train_labels)

print("\nEnter flower measurements:")
sepal_length = float(input("Sepal length: "))
sepal_width  = float(input("Sepal width: "))
petal_length = float(input("Petal length: "))
petal_width  = float(input("Petal width: "))

manual_input = np.array([sepal_length, sepal_width, petal_length, petal_width])

prediction = ppn.predict(manual_input)

if prediction == 1:
    print("\nPrediction: Iris-setosa ")
else:
    print("\nPrediction: Not Iris-setosa ")
from sklearn.metrics import accuracy_score

# Make predictions on the test set
y_preds = [ppn.predict(x) for x in test_data]

# Calculate accuracy
accuracy = accuracy_score(test_labels, y_preds)

# Print accuracy as percentage
print("\nTest Accuracy:", round(accuracy, 2) * 100, "%")