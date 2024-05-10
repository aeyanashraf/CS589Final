import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt

data = pd.read_csv('loan.csv')

data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})
data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})
data['Dependents'] = data['Dependents'].replace('3+', 3).astype(float)

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([-counts[i]/np.sum(counts) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def InfoGain(data, split_attribute_name, target_name="Loan_Status"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data[data[split_attribute_name]==vals[i]][target_name]) for i in range(len(vals))])
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain

def DecisionTree(data, features, target_attribute_name="Loan_Status", parent_node_class=None):
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
        
        tree = {best_feature:{}}
        
        features = [i for i in features if i != best_feature]

        for value in np.unique(data[best_feature]):
            sub_data = data[data[best_feature] == value]
            subtree = DecisionTree(sub_data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
            
        return tree

def predict(instance, tree):
    for nodes in tree.keys():
        value = instance[nodes]
        if value not in tree[nodes]:
            return np.random.choice([0, 1])
        tree = tree[nodes][value]
        if type(tree) is not dict:
            return tree
        else:
            prediction = predict(instance, tree)
    return prediction

def calculate_accuracy_f1(predictions, actual):
    tp = sum((predictions == 1) & (actual == 1))
    tn = sum((predictions == 0) & (actual == 0))
    fp = sum((predictions == 1) & (actual == 0))
    fn = sum((predictions == 0) & (actual == 1))
    
    accuracy = (tp + tn) / len(predictions)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, f1

def Random_Forest(data, number_of_trees, max_features=None):
    forest = []
    n_samples = len(data)
    for _ in range(number_of_trees):
        sample_data = data.sample(n=n_samples, replace=True)
        features = data.columns.tolist()
        features.remove('Loan_Status')
        if max_features is not None:
            features = np.random.choice(features, max_features, replace=False).tolist()
        tree = DecisionTree(sample_data, features)
        forest.append(tree)
    return forest

def random_forest_predict(instance, forest):
    predictions = [predict(instance, tree) for tree in forest]
    if len(predictions) == 1:
        return predictions[0]
    else:
        mode_result = mode(predictions)
        if isinstance(mode_result.mode, np.ndarray):
            majority_vote = mode_result.mode[0]
        else:
            majority_vote = mode_result.mode
        return majority_vote


def shuffle_split_data(data, k):
    # Shuffle the dataset
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    # Split data into k roughly equal parts
    n = len(data)
    return [data[i * n // k: (i + 1) * n // k] for i in range(k)]

def evaluate_random_forest_kfold(data, n_trees, k=10):
    results = []
    folds = shuffle_split_data(data, k)
    
    for i in range(k):
        train_data = pd.concat([folds[j] for j in range(k) if j != i]).reset_index(drop=True)
        test_data = folds[i].reset_index(drop=True)

        forest = Random_Forest(train_data, n_trees, max_features=int(np.sqrt(len(train_data.columns) - 1)))
        test_data['predictions'] = test_data.apply(random_forest_predict, axis=1, args=(forest,))
        accuracy, f1 = calculate_accuracy_f1(test_data['predictions'], test_data['Loan_Status'])
        results.append((accuracy, f1))
    
    average_accuracy = np.mean([acc for acc, _ in results])
    average_f1 = np.mean([f1 for _, f1 in results])
    
    return average_accuracy, average_f1

ntrees = [1, 5, 10, 20, 30, 40, 50]
forest_results = {n: evaluate_random_forest_kfold(data, n) for n in ntrees}

for n, (accuracy, f1) in forest_results.items():
    print(f"Trees: {n}, Average Accuracy: {accuracy:.2f}, Average F1-Score: {f1:.2f}")

tree_counts = [n for n in ntrees]
accuracies = [forest_results[n][0] for n in ntrees]
f1_scores = [forest_results[n][1] for n in ntrees]

plt.figure(figsize=(10, 5))
plt.plot(tree_counts, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
plt.plot(tree_counts, f1_scores, marker='o', linestyle='-', color='r', label='F1 Score')
plt.title('Random Forest Performance Evaluation')
plt.xlabel('Number of Trees')
plt.ylabel('Performance Metrics')
plt.xticks(tree_counts)
plt.legend()
plt.grid(True)
plt.show()