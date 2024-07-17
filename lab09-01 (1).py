#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install torch torchvision')


# In[ ]:


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# Define the transformations including normalization and flattening
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten the image to 1D array
])


# In[ ]:


# Load the MNIST training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Convert data to numpy arrays
train_data = train_dataset.data.numpy().reshape(-1, 28*28)
train_labels = train_dataset.targets.numpy()
test_data = test_dataset.data.numpy().reshape(-1, 28*28)
test_labels = test_dataset.targets.numpy()


# In[ ]:


# Reduce the size of the training data for faster computation
sample_size = 10000  # Choose a smaller sample size
indices = np.random.choice(len(train_data), sample_size, replace=False)
train_data_sample = train_data[indices]
train_labels_sample = train_labels[indices]


# In[ ]:


# Cross-validation to find the best value of k
def cross_validate(k_values, X_train, y_train):
    kf = KFold(n_splits=5)
    accuracies = {k: [] for k in k_values}

    for train_index, val_index in kf.split(X_train):
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')  # Using L1 distance
            knn.fit(X_tr, y_tr)
            y_pred = knn.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            accuracies[k].append(accuracy)

    avg_accuracies = {k: np.mean(v) for k, v in accuracies.items()}
    return avg_accuracies

k_values = [1, 2, 5, 10, 20]
accuracies = cross_validate(k_values, train_data_sample, train_labels_sample)


# In[ ]:


# Plotting accuracies
plt.figure()
plt.plot(k_values, [accuracies[k] for k in k_values])
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('5-Fold Cross-Validation Accuracy for Different k')
plt.show()

best_k = max(accuracies, key=accuracies.get)
print(f'Best value of k: {best_k}')


# In[ ]:


# Evaluate on the test set using the best k
knn = KNeighborsClassifier(n_neighbors=best_k, metric='manhattan')
knn.fit(train_data, train_labels)
test_preds = knn.predict(test_data)

accuracy = accuracy_score(test_labels, test_preds)
precision = precision_score(test_labels, test_preds, average='macro')
recall = recall_score(test_labels, test_preds, average='macro')
f1 = f1_score(test_labels, test_preds, average='macro')
conf_matrix = confusion_matrix(test_labels, test_preds)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')

