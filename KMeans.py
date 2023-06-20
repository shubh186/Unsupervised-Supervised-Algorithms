import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import mode

# loading dataset 
df = pd.read_csv('iris.csv')

# random train-test split
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)
X_train = train_df.drop('Species', axis=1)
y_train = train_df['Species']
X_test = test_df.drop('Species', axis=1)
y_test = test_df['Species']

# preprocessing the data
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# determine the optimal number of clusters (K)
wcss = []  # create an empty list to store the within-cluster sum of squares (WCSS) for each k
for i in range(1, 11):  # iterate over a range of k values from 1 to 10
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)  # create a KMeans object with i clusters
    kmeans.fit(X_train)  # fit the KMeans object to the training data
    wcss.append(kmeans.inertia_)  # add the WCSS of the current KMeans object to the list of WCSS values

# run K-Means algorithm multiple times
# Part 6: Record distortion of each run and pick best run
N = 100
distortions = []
for i in range(N):
    kmeans = KMeans(n_clusters=3, init='random', max_iter=300, n_init=10, random_state=i)
    kmeans.fit(X_train)
    distortions.append(kmeans.inertia_)
best_kmeans = KMeans(n_clusters=3, init='random', max_iter=300, n_init=10, random_state=distortions.index(min(distortions)))

# Part 7: Assign cluster labels to training set and evaluate accuracy
y_pred_train = best_kmeans.fit_predict(X_train)
cluster_labels_train = np.zeros_like(y_pred_train)
for i in range(3):
    mask = (y_pred_train == i)
    cluster_labels_train[mask] = mode(y_train[mask], keepdims=True)[0]
accuracy_train = accuracy_score(y_train, cluster_labels_train)

# Apply K-Means to testing set and evaluate accuracy
y_pred_test = best_kmeans.predict(X_test)
cluster_labels_test = np.zeros_like(y_pred_test)
for i in range(3):
    mask = (y_pred_test == i)
    cluster_labels_test[mask] = mode(y_test[mask], keepdims=True)[0]
accuracy_test = accuracy_score(y_test, cluster_labels_test)

# Print accuracy percentages for each class on training and testing sets
target_names = le.classes_
for i, target_name in enumerate(target_names):
    mask_train = (cluster_labels_train == i)
    mask_test = (cluster_labels_test == i)
    accuracy_class_train = accuracy_score(y_train[mask_train], cluster_labels_train[mask_train])
    accuracy_class_test = accuracy_score(y_test[mask_test], cluster_labels_test[mask_test])
    print(f"{target_name} accuracy: = {accuracy_class_test:.0%}")

