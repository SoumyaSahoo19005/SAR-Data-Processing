#!/usr/bin/env python
# coding: utf-8

# In[72]:


# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[79]:


# Load the data
data = pd.read_csv('LAI.csv')
# Separate features and target
X = data.drop(["LAI"],axis=1)
y = data["LAI"]


# In[87]:


data


# In[81]:


# Standardize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)


# In[82]:


# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)


# In[83]:


# Print explained variance ratio
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")


# In[86]:


# Create new dataframe with PCA components and target variable
pca_df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
pca_df["actual_output_column_name"] = y


# In[92]:


A = pca_df.iloc[:, 0:2]
B = pca_df.iloc[:, -1]


# In[102]:


scaler = MinMaxScaler()
A_scaled = scaler.fit_transform(A)
A_train, A_test, B_train, B_test = train_test_split(A_scaled, B, test_size=0.3, random_state=42)


# In[103]:


reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(A_train, B_train)


# In[104]:


score = reg.score(A_test, B_test)
print(f"R-squared score: {score:.3f}")


# In[65]:


# Load the data
data = pd.read_csv('LAI.csv')
data = data.reset_index(drop=True)
# Split the data into training and testing sets
X = data.iloc[:, 0:6]
y = data.iloc[:, -1]


# In[109]:





# In[67]:


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# In[68]:


reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)


# In[69]:


score = reg.score(X_test, y_test)
print(f"R-squared score: {score:.3f}")


# In[107]:





# In[ ]:


import numpy as np
from sklearn.metrics import mean_squared_error

# Calculate MSE
mse = mean_squared_error(y_true, y_pred)

# Calculate RMSE
rmse = np.sqrt(mse)

print("MSE:", mse)
print("RMSE:", rmse)


# In[ ]:




