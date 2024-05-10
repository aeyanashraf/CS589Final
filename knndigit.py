from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
digits = datasets.load_digits(return_X_y=True)
X = digits[0]
y = digits[1]

def euclidean_distance(a, b):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt(np.sum((a - b) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values."""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Return the predicted labels for X."""
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return np.bincount(k_nearest_labels).argmax()

def f1_score_from_scratch(y_true, y_pred):
    unique_classes = np.unique(y_true)
    precision = {}
    recall = {}
    f1_scores = {}
    
    for label in unique_classes:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        
        precision[label] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[label] = tp / (tp + fn) if (tp + fn) > 0 else 0
        if precision[label] + recall[label] > 0:
            f1_scores[label] = 2 * (precision[label] * recall[label]) / (precision[label] + recall[label])
        else:
            f1_scores[label] = 0
    
    weights = [np.sum(y_true == label) for label in unique_classes]
    weighted_f1 = sum(f1_scores[label] * weight for label, weight in zip(unique_classes, weights)) / sum(weights)
    return weighted_f1

def k_fold_cross_validation(X, y, k_folds, ks):
    fold_size = len(X) // k_folds
    results = []

    for k in ks:
        accuracies = []
        f1_scores = []

        for i in range(k_folds):
            start, end = i * fold_size, (i + 1) * fold_size
            X_val, y_val = X[start:end], y[start:end]
            X_train = np.concatenate((X[:start], X[end:]))
            y_train = np.concatenate((y[:start], y[end:]))

            model = KNN(k=k)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            accuracy = np.mean(predictions == y_val)
            f1 = f1_score_from_scratch(y_val, predictions)

            accuracies.append(accuracy)
            f1_scores.append(f1)

        results.append((k, np.mean(accuracies), np.mean(f1_scores)))

    return results

# Define the list of k values
ks = [1, 5, 10, 20, 30, 40, 50]

# Perform k-fold cross-validation and evaluate the classifier
results = k_fold_cross_validation(X, y, 10, ks)
# Extracting results for plotting
ks, accuracies, f1_scores = zip(*results)

print("k-NN performance results:")
for k, accuracy, f1_score in zip(ks, accuracies, f1_scores):
    print(f"k={k}: Accuracy={accuracy:.3f}, F1-Score={f1_score:.3f}")

    
# Plotting results
plt.figure(figsize=(10, 5))
plt.plot(ks, accuracies, label='Accuracy', marker='o')
plt.plot(ks, f1_scores, label='F1 Score', marker='o')
plt.xlabel('Number of Neighbors: k')
plt.ylabel('Performance')
plt.title('kNN Performance with Different k')
plt.legend()
plt.grid(True)
plt.show()
