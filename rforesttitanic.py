import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('titanic.csv')
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}).astype(int)
data['Age'].fillna(data['Age'].mean(), inplace=True)

features = ['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']
target_attribute_name = 'Survived'

def bootstrap_sample(data, n_samples):
    return data.sample(n=n_samples, replace=True)

def InfoGain(data, split_attribute_name, target_name="target"):

    total_entropy = entropy(data[target_name])

    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    return -np.sum([(count / np.sum(counts)) * np.log2(count) for count in counts if count > 0])

def DecisionTree(data, features, target_attribute_name='Survived', parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0 or len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = data[target_attribute_name].mode()[0]

        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]

        for value in np.unique(data[best_feature]):
            sub_data = data[data[best_feature] == value]
            subtree = DecisionTree(sub_data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree

        return tree

def predict(row, tree):
    while isinstance(tree, dict):
        split_attribute = next(iter(tree))  
        tree = tree[split_attribute]
        key = row[split_attribute]
        if key in tree:
            tree = tree[key]
        else:
            return np.nan 
    return tree

def build_random_forest(data, features, n_trees, sample_size, feature_subset_size=None):
    forest = []
    for _ in range(n_trees):
        sample = bootstrap_sample(data, sample_size)
        if feature_subset_size:
            sampled_features = np.random.choice(features, size=feature_subset_size, replace=False).tolist()
        else:
            sampled_features = features
        tree = DecisionTree(sample, sampled_features)
        forest.append(tree)
    return forest

def random_forest_predict(forest, row):
    predictions = [predict(row, tree) for tree in forest]
    predictions = [p for p in predictions if p == p]
    if len(predictions) == 0:
        return np.nan
    return max(set(predictions), key=predictions.count)

def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

def f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def evaluate_forest(forest, data):
    data['prediction'] = data.apply(lambda x: random_forest_predict(forest, x), axis=1)
    valid_data = data.dropna(subset=['prediction'])
    acc = accuracy(valid_data['Survived'], valid_data['prediction'])
    f1 = f1_score(valid_data['Survived'], valid_data['prediction'])
    return acc, f1

def k_fold_split(data, k):
    data_shuffled = data.sample(frac=1).reset_index(drop=True)
    folds = np.array_split(data_shuffled, k)  
    return folds

def k_fold_cross_validation(data, features, n_trees, k=10, feature_subset_size=None):
    folds = k_fold_split(data, k)
    acc_scores = []
    f1_scores = []
    
    for i in range(k):
        test_data = folds[i]
        train_data = pd.concat([folds[j] for j in range(k) if j != i], ignore_index=True)
        forest = build_random_forest(train_data, features, n_trees, len(train_data), feature_subset_size)
        test_data['prediction'] = test_data.apply(lambda x: random_forest_predict(forest, x), axis=1)
        valid_test_data = test_data.dropna(subset=['prediction']) 
        acc = accuracy(valid_test_data['Survived'], valid_test_data['prediction'])
        f1 = f1_score(valid_test_data['Survived'], valid_test_data['prediction'])
        
        acc_scores.append(acc)
        f1_scores.append(f1)
    
    return np.mean(acc_scores), np.mean(f1_scores)

def analyze_forest_performance_with_kfold(data, features, trees_list, k=10, feature_subset_size=None):
    results = []
    for n_trees in trees_list:
        acc, f1 = k_fold_cross_validation(data, features, n_trees, k, feature_subset_size)
        results.append((n_trees, acc, f1))
        print(f"Number of Trees: {n_trees}, Average Accuracy: {acc:.4f}, Average F1-Score: {f1:.4f}")
    return results


ntrees_list = [1, 5, 10, 20, 30, 40, 50]
feature_subset_size = int(np.sqrt(len(features)))
results = analyze_forest_performance_with_kfold(data, features, ntrees_list, k=10, feature_subset_size=feature_subset_size)

# Extract the number of trees, average accuracy, and average F1 scores from the results
ntrees_list = [result[0] for result in results]
average_accuracies = [result[1] for result in results]
average_f1_scores = [result[2] for result in results]

plt.figure(figsize=(10, 5))
plt.plot(ntrees_list, average_accuracies, label='Average Accuracy', marker='o', color='blue')
plt.plot(ntrees_list, average_f1_scores, label='Average F1 Score', marker='x', color='red')
plt.title('Performance of Random Forest with Varying Number of Trees (Titanic Dataset)')
plt.xlabel('Number of Trees')
plt.ylabel('Performance Metrics')
plt.legend()
plt.grid(True)
plt.show()