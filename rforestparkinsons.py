import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

data = pd.read_csv('parkinsons.csv')

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([-counts[i] / np.sum(counts) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy

def InfoGain(data, split_attribute_name, target_name="Diagnosis"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name]) for i in range(len(vals))])
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain

def DecisionTree(data, features, target_attribute_name="Diagnosis", parent_node_class=None, n_features=None):
    if n_features is None:
        n_features = len(features)
    else:
        n_features = min(n_features, len(features))
    if n_features < len(features):
        features = random.sample(features, n_features)

    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0 or len(features) == 0:
        majority_class_index = np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])
        return np.unique(data[target_attribute_name])[majority_class_index]
    else:
        majority_class_index = np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])
        parent_node_class = np.unique(data[target_attribute_name])[majority_class_index]
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = DecisionTree(sub_data, features, target_attribute_name, parent_node_class, n_features)
            tree[best_feature][value] = subtree
        return tree

def predict(instance, tree):
    for nodes in tree.keys():
        value = instance[nodes]
        if value not in tree[nodes]:
            return np.random.choice([0, 1])
        tree = tree[nodes][value]
        if isinstance(tree, dict):
            return predict(instance, tree)
        else:
            return tree

def RandomForest(data, n_trees, n_features=None):
    trees = []
    for _ in range(n_trees):
        sample = data.sample(n=len(data), replace=True)
        features = data.columns[:-1] 
        tree = DecisionTree(sample, features.tolist(), n_features=n_features)
        trees.append(tree)
    return trees

def RandomForest_predict(trees, instance):
    predictions = [predict(instance, tree) for tree in trees]
    return max(set(predictions), key=predictions.count) 


def cross_validation(data, n_trees, n_features=None, folds=10):
    fold_size = len(data) // folds
    accuracies = []
    f1_scores = []

    trees = RandomForest(data, n_trees, n_features=n_features)

    for i in range(folds):
        test_data = data.iloc[i * fold_size:(i + 1) * fold_size]
        predictions = test_data.apply(lambda row: RandomForest_predict(trees, row), axis=1)
        true_labels = test_data.iloc[:, -1]
        accuracy = np.sum(predictions == true_labels) / len(true_labels)
        accuracies.append(accuracy)

        tp = np.sum((predictions == 1) & (true_labels == 1))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1_score)

    return np.mean(accuracies), np.mean(f1_scores)

tree_counts = [1, 5, 10, 20, 30, 40, 50]
results = []
for count in tree_counts:
    average_accuracy, average_f1_score = cross_validation(data, n_trees=count, n_features=int(np.sqrt(len(data.columns)-1)))
    results.append((count, average_accuracy, average_f1_score))

for result in results:
    print(f"Trees: {result[0]}, Average Accuracy: {result[1]:.4f}, Average F1 Score: {result[2]:.4f}")

tree_counts = [1, 5, 10, 20, 30, 40, 50]
average_accuracies = [accuracy for _, accuracy, _ in results]
average_f1_scores = [f1 for _, _, f1 in results]

plt.figure(figsize=(10, 5))
plt.plot(tree_counts, average_accuracies, label='Average Accuracy', marker='o', color='blue')
plt.plot(tree_counts, average_f1_scores, label='Average F1 Score', marker='x', color='red')
plt.title('Performance of Random Forest with Varying Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Performance Metrics')
plt.legend()
plt.grid(True)
plt.show()