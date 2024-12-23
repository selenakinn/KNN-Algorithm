# Installing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Loading the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

# Dataset properties and target variable
X = iris.data  # Features (Sepal length, Sepal width, Petal length, Petal width)
y = iris.target  # Types (Setosa, Versicolor, Virginica)

# Splitting into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the KNN model
k = 3  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)

# Training the model
knn.fit(X_train, y_train)

# Prediction on test data
y_pred = knn.predict(X_test)

# Model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluğu: {accuracy * 100:.2f}%")

# A new data estimate
new_data = [[4.9, 3.6, 1.4, 0.2]] #Sepal length, Sepal width, Petal length, Petal width
prediction = knn.predict(new_data)
print(f"Yeni Veri Tahmini: {iris.target_names[prediction[0]]}")
