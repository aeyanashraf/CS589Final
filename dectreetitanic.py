import pandas as pd
import numpy as np

data = pd.read_csv('titanic.csv')
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}).astype(int)
data['Age'].fillna(data['Age'].mean(), inplace=True) 

features = ['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']
target = 'Survived'

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

def k_fold_split(data, k):
    data_shuffled = data.sample(frac=1).reset_index(drop=True) 
    folds = np.array_split(data_shuffled, k) 
    return folds

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

def k_fold_cross_validation(data, k, features, target_attribute_name):
    folds = k_fold_split(data, k)
    acc_scores = []
    f1_scores = []
    
    for i in range(k):
        test_data = folds[i]
        train_data = pd.concat([folds[j] for j in range(k) if j != i], ignore_index=True)
        tree = DecisionTree(train_data, features, target_attribute_name)

        test_data['prediction'] = test_data.apply(lambda x: predict(x, tree), axis=1)
        

        valid_test_data = test_data.dropna(subset=['prediction'])
        true_labels = valid_test_data[target_attribute_name]
        predictions = valid_test_data['prediction']
        
        acc_scores.append(accuracy(true_labels, predictions))
        f1_scores.append(f1_score(true_labels, predictions))
    
    return np.mean(acc_scores), np.mean(f1_scores)

features = ['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']
target_attribute_name = 'Survived'

average_accuracy, average_f1_score = k_fold_cross_validation(data, 10, features, target_attribute_name)
print("Average Accuracy:", average_accuracy)
print("Average F1-Score:", average_f1_score)
