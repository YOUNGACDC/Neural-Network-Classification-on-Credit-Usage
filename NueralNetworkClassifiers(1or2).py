# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:10:00 2024

@author: Armanis
"""
#%% Import required packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#%% Define the data (Years, Salary, Used Credit)
data = {
    'Years': [4, 18, 1, 3, 15, 6],
    'Salary': [43, 65, 53, 95, 88, 112],
    'Used Credit': [0, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

#%% Split data into features (X) and target (y)
X = df[['Years', 'Salary']]  # Features
y = df['Used Credit']       # Target

#%% Scale the data to the range [0, 1] using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # Scale the features

#%% Split the data into training and validation sets (60% training, 40% validation)
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.4, random_state=12)

#%% Train a neural network model for classification using MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(5), activation='logistic', solver='lbfgs', random_state=1)
model.fit(X_train, y_train)

#%% Predict on the validation set
y_pred = model.predict(X_valid)

#%% Evaluate the model's performance using confusion matrix and classification report
cm = confusion_matrix(y_valid, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification report for more detailed performance evaluation
print("\nClassification Report:")
print(classification_report(y_valid, y_pred))

#%% Plot the confusion matrix for better visualization
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Credit', 'Used Credit'], yticklabels=['No Credit', 'Used Credit'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#%% Accuracy of the model
accuracy = model.score(X_valid, y_valid)
print(f"Validation Accuracy: {accuracy:.2f}")

#The key ingredient for a neural network's ability to evolve toward more accurate predictions is backpropagation combined with gradient descent. 