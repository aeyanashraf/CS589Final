import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
df = pd.DataFrame(digits.data)
df['target'] = digits.target
features = df.columns[:-1]

def bootstrap_sample(data):
    return data.sample(n=len(data), replace=True)

def feature_subset(features, subset_size):
    return np.random.choice(features, size=subset_size, replace=False)

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([-counts[i]/np.sum(counts) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def InfoGain(data, split_attribute_name, target_name="target"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = np.sum([
        (counts[i]/np.sum(counts)) * entropy(data[data[split_attribute_name]==vals[i]][target_name])
        for i in range(len(vals))
    ])
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain

def DecisionTree(data, features, target_attribute_name="target", parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0 or len(features) == 0:
        majority_class_index = np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])
        return np.unique(data[target_attribute_name])[majority_class_index]
    else:
        majority_class_index = np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])
        parent_node_class = np.unique(data[target_attribute_name])[majority_class_index]
        
        item_values = [InfoGain(data, feature) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        tree = {best_feature:{}}
        features = [i for i in features if i != best_feature]
        
        for value in np.unique(data[best_feature]):
            sub_data = data[data[best_feature] == value]
            subtree = DecisionTree(sub_data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
            
        return tree
    
def build_forest(data, features, n_trees, subset_size):
    forest = []
    for _ in range(n_trees):
        sample = bootstrap_sample(data)
        subset_features = feature_subset(features, subset_size)
        tree = DecisionTree(sample, subset_features)
        forest.append(tree)
    return forest

def accuracy_score(true_labels, predictions):
    correct = np.sum(true_labels == predictions)
    total = len(true_labels)
    return correct / total

def precision_recall_f1(true_labels, predictions):
    unique_classes = np.unique(true_labels)
    precision = {}
    recall = {}
    f1 = {}
    
    for cls in unique_classes:
        tp = np.sum((predictions == cls) & (true_labels == cls))
        fp = np.sum((predictions == cls) & (true_labels != cls))
        fn = np.sum((predictions != cls) & (true_labels == cls))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precision[cls] = prec
        recall[cls] = rec
        f1[cls] = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    macro_f1 = np.mean(list(f1.values()))
    return macro_f1

def classify(tree, instance):
    if not isinstance(tree, dict):
        return tree
    attribute_index = list(tree.keys())[0]
    subtree = tree[attribute_index].get(instance[attribute_index], np.random.choice(list(tree[attribute_index].values())))
    return classify(subtree, instance)

def random_forest_predict(forest, instance):
    predictions = [classify(tree, instance) for tree in forest]
    return max(set(predictions), key=predictions.count)

def evaluate_forest(forest, data):
    predictions = data.apply(lambda x: random_forest_predict(forest, x), axis=1)
    true_labels = data['target']
    f1 = precision_recall_f1(true_labels, predictions)
    accuracy = accuracy_score(true_labels, predictions)
    return f1, accuracy

def analyze_forest(data, features, trees_list):
    results = []
    for n_trees in trees_list:
        forest = build_forest(data, features, n_trees, int(np.sqrt(len(features))))
        f1, accuracy = evaluate_forest(forest, data)
        results.append((n_trees, f1, accuracy))
    return results

trees_list = [1, 5, 10, 20, 30, 40, 50]
results = analyze_forest(df, features, trees_list)

for n_trees, f1, accuracy in results:
    print(f"Number of Trees: {n_trees}, Average F1 Score: {f1:.4f}, Average Accuracy: {accuracy:.4f}")

f1_scores = [f1 for _, f1, _ in results]
accuracies = [acc for _, _, acc in results]

plt.figure(figsize=(10, 5))
plt.plot(trees_list, f1_scores, label='F1 Score', marker='o', color='blue')
plt.plot(trees_list, accuracies, label='Accuracy', marker='x', color='red')
plt.title('Random Forest Performance Evaluation')
plt.xlabel('Number of Trees')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
plt.show()