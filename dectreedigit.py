import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

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

digits = load_digits()
df = pd.DataFrame(digits.data)
df['target'] = digits.target
features = df.columns[:-1]

def classify(tree, instance):
    if not isinstance(tree, dict):
        return tree
    attribute_index = list(tree.keys())[0]
    subtree = tree[attribute_index].get(instance[attribute_index], np.random.choice(list(tree[attribute_index].values())))
    return classify(subtree, instance)

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

def k_fold_cross_validation(data, k=10):
    kf_indices = np.array_split(np.random.permutation(len(data)), k)
    accuracies = []
    f1_scores = []

    for i in range(k):
        test_indices = kf_indices[i]
        train_indices = np.concatenate([kf_indices[j] for j in range(k) if j != i])
        
        train_data = data.iloc[train_indices]
        test_data = data.iloc[test_indices]
        
        tree = DecisionTree(train_data, list(features))
        predictions = test_data.apply(lambda x: classify(tree, x), axis=1)
        true_labels = test_data['target']
        
        f1 = precision_recall_f1(true_labels, predictions)
        accuracy = accuracy_score(true_labels, predictions)
        
        accuracies.append(accuracy)
        f1_scores.append(f1)

    return np.mean(f1_scores), np.mean(accuracies)

mean_f1, mean_accuracy = k_fold_cross_validation(df)

print(f"Average F1 Score: {mean_f1:.4f}")
print(f"Average Accuracy: {mean_accuracy:.4f}")
