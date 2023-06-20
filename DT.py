import csv
import random
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

MAX_TREE_DEPTH = 3
MINIMUM_SAMPLE_SIZE = 4

class TreeNode:

    def __init__(self, training_set, attribute_list, attribute_values, tree_depth):
        self.is_leaf = False
        self.dataset = training_set
        self.split_attribute = None
        self.split = None
        self.attribute_list = attribute_list
        self.attribute_values = attribute_values
        self.left_child = None
        self.right_child = None
        self.prediction = None
        self.depth = tree_depth

    def build(self):

        training_set = self.dataset

        if (self.depth >= MAX_TREE_DEPTH or len(training_set) < MINIMUM_SAMPLE_SIZE or len(set([elem["Species"] for elem in training_set])) <= 1):
            # Leaf node
            self.is_leaf = True
            if self.is_leaf:
                self.prediction = max(set([elem["Species"] for elem in training_set]), key=[elem["Species"] for elem in training_set].count)
        else:
            # Find best attribute to split on
            max_gain, attribute, split = max_information_gain(self.attribute_list, self.attribute_values, training_set)
            if max_gain > 0:
                # Split node
                self.split = split
                self.split_attribute = attribute
                training_set_l = [elem for elem in training_set if elem[attribute] < split]
                training_set_r = [elem for elem in training_set if elem[attribute] >= split]
                # Create children
                self.left_child = TreeNode(training_set_l, self.attribute_list, self.attribute_values, self.depth + 1)
                self.right_child = TreeNode(training_set_r, self.attribute_list, self.attribute_values, self.depth + 1)
                self.left_child.build()
                self.right_child.build()
            else:
                # Leaf node
                self.is_leaf = True
                if self.is_leaf:
                    self.prediction = max(set([elem["Species"] for elem in training_set]), key=[elem["Species"] for elem in training_set].count)
    # test decision tree accuracy
    
    def predict(self, sample):
        if self.is_leaf:
            return self.prediction
        else:
            if sample[self.split_attribute] < self.split:
                return self.left_child.predict(sample)
            else:
                return self.right_child.predict(sample)


class ID3_tree:
    def __init__(self):
        self.root = None

    def build(self, training_set, attribute_list, attribute_values):
        self.root = TreeNode(training_set, attribute_list, attribute_values, 0)
        self.root.build()

    def predict(self, sample):
        return self.root.predict(sample)

# calculate the entropy of a target_attribute for a given set
def entropy(dataset):
    if len(dataset) == 0:
        return 0
    target_attribute_name = "Species"
    target_attribute_values = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    data_entropy = 0
    for val in target_attribute_values:

        # calculate the probability p that an element in the set has the value val
        p = len([elem for elem in dataset if elem[target_attribute_name] == val]) / len(dataset)

        if p > 0:
            data_entropy += -p * math.log(p, 2)

    return data_entropy

def info_gain(attribute_name, split, dataset):

    # split set and calculate probabilities that elements are in the splits
    set_smaller = [elem for elem in dataset if elem[attribute_name] < split]
    p_smaller = len(set_smaller) / len(dataset)
    set_greater_equals = [elem for elem in dataset if elem[attribute_name] >= split]
    p_greater_equals = len(set_greater_equals) / len(dataset)

    # calculate information gain
    info_gain = entropy(dataset)
    info_gain -= p_smaller * entropy(set_smaller)
    info_gain -= p_greater_equals * entropy(set_greater_equals)

    return info_gain

# get criterion and optimal split to maximize information gain
def max_information_gain(attribute_list, attribute_values, dataset):

    max_info_gain = 0
    for attribute in attribute_list:  # test all input attributes
        for split in attribute_values[attribute]:  # test all possible values as split limits
            split_info_gain = info_gain(attribute, split, dataset)  # calculate information gain
            if split_info_gain >= max_info_gain:
                max_info_gain = split_info_gain
                max_info_gain_attribute = attribute
                max_info_gain_split = split
    return max_info_gain, max_info_gain_attribute, max_info_gain_split


def read_iris_dataset():
    dataset = []
    with open('iris.csv', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        is_first = True
        for row in reader:
            instance = []
            if not is_first:
                # skip first row
                instance.append(float(row[1]))
                instance.append(float(row[2]))
                instance.append(float(row[3]))
                instance.append(float(row[4]))
                instance.append(row[5])
                dataset.append(instance)
            is_first = False
    return dataset

if __name__ == '__main__':
    # load iris dataset
    iris = read_iris_dataset()

    # randomly select 120 rows for training, and use the remaining 30 rows for testing
    np.random.seed(42)
    indices = np.random.permutation(len(iris))
    X_train = np.array([iris[i][:-1] for i in indices[:120]])
    y_train = np.array([iris[i][-1] for i in indices[:120]])
    X_test = np.array([iris[i][:-1] for i in indices[120:]])
    y_test = np.array([iris[i][-1] for i in indices[120:]])

    # list of all input attributes
    attr_list = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

    # get list of all valid attribute values
    # this will later be needed to calculate the information gain
    attr_domains = {}
    for attr in attr_list:
        attr_domain = set()
        for s in X_train:
            attr_domain.add(s[attr_list.index(attr)])
        attr_domains[attr] = list(attr_domain)

    # build decision tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)

    # evaluate accuracy on testing set for each class
    classes = set(y_test)
    for c in classes:
        c_indices = np.where(y_test == c)
        c_X_test = X_test[c_indices]
        c_y_test = y_test[c_indices]
        accuracy = dt.score(c_X_test, c_y_test)
        print("{} accuracy: {:.2f}%".format(c, accuracy * 100))
