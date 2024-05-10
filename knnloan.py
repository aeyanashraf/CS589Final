import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('loan.csv')

# Preprocessing
def preprocess_data(data):
    # Convert categorical variables into numeric
    categorical_vars = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    for var in categorical_vars:
        data[var] = pd.Categorical(data[var]).codes  # Convert to category codes; handle NaN as -1

    # Convert 'Dependents' to numeric; treat "3+" as 3
    data['Dependents'] = data['Dependents'].replace('3+', 3).astype(float)

    # Handle missing numerical data
    num_vars = ['Dependents', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    for var in num_vars:
        data[var].fillna(data[var].mean(), inplace=True)  # Impute with the mean

    # Select features and target
    features = data[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                     'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                     'Credit_History', 'Property_Area']]
    target = data['Loan_Status']

    return features.values, target.values

X, y = preprocess_data(data)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
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

def manual_k_fold_cross_validation(X, y, k_folds, ks):
    fold_size = len(X) // k_folds
    results = []

    for k in ks:
        accuracies = []
        f1_scores = []

        for fold in range(k_folds):
            start, end = fold * fold_size, (fold + 1) * fold_size if fold != k_folds - 1 else len(X)
            X_train = np.concatenate((X[:start], X[end:]))
            y_train = np.concatenate((y[:start], y[end:]))
            X_val = X[start:end]
            y_val = y[start:end]

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
results = manual_k_fold_cross_validation(X, y, 10, ks)

# Extracting results for plotting
ks, accuracies, f1_scores = zip(*results)

# Print formatted results
print("k-NN performance results:")
for k, accuracy, f1_score in zip(ks, accuracies, f1_scores):
    print(f"k={k}: Accuracy={accuracy:.3f}, F1-Score={f1_score:.3f}")

# Plotting results
plt.figure(figsize=(10, 5))
plt.plot(ks, accuracies, label='Accuracy', marker='o')
plt.plot(ks, f1_scores, label='F1 Score', marker='o')
plt.xlabel('Number of Neighbors: k')
plt.ylabel('Performance')
plt.title('kNN Performance with Different k on Loan Dataset')
plt.legend()
plt.grid(True)
plt.show()
