#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# In[2]:


#Random Forest
# Load the data
data = pd.read_csv('LEVEL1.csv')
data = data.reset_index(drop=True)
# Split the data into training and testing sets
X = data.iloc[:, 1:7]
y = data.iloc[:, -1]

# assume you have a numpy array of input data named 'X'
# instantiate the scaler object
scaler = MinMaxScaler()
# fit the scaler to the input data and transform the data
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

# Train the model for RF
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
# Predict on the test data
y_pred = rf.predict(X_test)
# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Create confusion matrix
matrix = confusion_matrix(y_test, y_pred)

# Print confusion matrix
print(matrix)
# Get class-wise accuracy score
report = classification_report(y_test, y_pred)

# Print accuracy report
print(report)


# In[4]:


n=100
#n_estimators = [10,11,12,13,14,15,20,25, 30, 40, 50]
Accuracy_list = []


# In[5]:


for i in range(1,n+1):
    rf = RandomForestClassifier(n_estimators=i, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    Accuracy_list.append(accuracy)


# In[6]:


fig, ax = plt.subplots()
ax.plot(n_estimators, Accuracy_list, label='Accuracy_Score')
ax.set_xlabel('Number of Trees')
ax.set_ylabel('Accuracy')
ax.set_title('LEVEL 1 RANDOM FOREST(Accuracy vs Number of Trees)')
ax.legend()
plt.show()


# In[7]:


# Create confusion matrix
matrix = confusion_matrix(y_test, y_pred)

# Print confusion matrix
print(matrix)
# Get class-wise accuracy score
report = classification_report(y_test, y_pred)

# Print accuracy report
print(report)


# In[7]:


#svm classifier
# Train the model for SVM

#randome_state=[32,42,52,62,72,82,92]
svm = SVC(kernel='poly', random_state=42)
svm.fit(X_train, y_train)
# Predict on the test data
y_pred = svm.predict(X_test)
# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Create confusion matrix
matrix = confusion_matrix(y_test, y_pred)

# Print confusion matrix
print(matrix)
# Get class-wise accuracy score
report = classification_report(y_test, y_pred)

# Print accuracy report
print(report)


# In[8]:


kernel1=['linear','poly','sigmoid']
Accuracy_list = []
for n in kernel1:
    print(n)


# In[9]:


for n in kernel1:
    svm = SVC(kernel=n, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    Accuracy_list.append(accuracy)
    


# In[12]:


fig, ax = plt.subplots()
ax.plot(kernel1, Accuracy_list, label='Accuracy_Score')
ax.set_xlabel('Kernel')
ax.set_ylabel('Accuracy')
ax.set_title('LEVEL 1 SVM(Accuracy vs Kernel)')
ax.legend()
plt.show()


# In[ ]:





# In[3]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[4]:


# Load the data
data = pd.read_csv('LEVEL1.csv')
data = data.reset_index(drop=True)
# Split the data into training and testing sets
X = data.iloc[:, 1:7].values
y = data.iloc[:, -1].values
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


# In[5]:


from sklearn.preprocessing import StandardScaler

# Scale the input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[8]:


model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(6,)),
    layers.Dense(16, activation='relu'),
    #layers.Dense(8, activation='relu')
    layers.Dense(4, activation='softmax')
])


# In[8]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[9]:


model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))


# In[ ]:


test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)


# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[30]:





# In[ ]:





# In[36]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




