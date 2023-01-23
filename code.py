"""
Inputs:
    X: n*m  numerical matrix which "n" is the number of samples, and "m" is the number of features
    y: n*1  array which has the labels of rows in X
"""

# DT Visualization =========================== Behzad Amanpour ==========================
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt

model1 = DecisionTreeClassifier(random_state=21)  # for reproducibility of the result, "random_state" has to be fixed to an integer
model1.fit(X, y)
fig = plt.figure(figsize=(18,18))
tree.plot_tree(model1, 
               feature_names=['name1', 'name2', ..., 'name n'],
               class_names=np.array(['class name1', 'class name2']),
               filled=True)

# Regularization (Pruning) =================== Behzad Amanpour =========================
model2 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, max_leaf_nodes=10, random_state=21) 
model2.fit(X, y)
fig = plt.figure(figsize=(18,18))
tree.plot_tree(model2,
               feature_names=['name1', 'name2', ..., 'name n'],
               class_names=np.array(['class name1', 'class name2']),
               filled=True)

# Cross-validation =========================== Behzad Amanpour =========================
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model1, X, y, cv=5, scoring='f1') # scoring could be 'recall', 'precision', 'accuracy', ... 
print("cross-val f1:", np.mean(scores))
scores = cross_val_score(model2, X, y, cv=5, scoring='f1')
print("cross-val f1:", np.mean(scores))
