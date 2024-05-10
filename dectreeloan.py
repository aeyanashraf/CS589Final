import numpy as np
import pandas as pd

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

def k_fold_cross_validation(data, k=10):
    folds = np.array_split(data, k)
    results = []
    for i in range(k):
        train = pd.concat([folds[j] for j in range(k) if j != i])
        test = folds[i]
        features = train.columns.tolist()
        features.remove('Loan_Status')
        tree = DecisionTree(train, features)
        predictions = test.apply(predict, axis=1, args=(tree,))
        actual = test['Loan_Status']
        accuracy, f1 = calculate_accuracy_f1(predictions, actual)
        results.append((accuracy, f1))
    return results

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

results = k_fold_cross_validation(data, k=10)
average_accuracy = np.mean([acc for acc, f1 in results])
average_f1 = np.mean([f1 for acc, f1 in results])
print(f"Average Accuracy: {average_accuracy:.2f}, Average F1-Score: {average_f1:.2f}")
