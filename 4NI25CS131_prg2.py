#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
df = pd.read_csv(r"C:\Users\user\Downloads\archive (9)\winequality-red.csv")
print("First five rows:")
print(df.head())


# In[7]:


print("\nDataset shape:")
print(df.shape)


# In[9]:


print("\nColumn names:")
print(df.columns)


# In[10]:


print("\nMissing values:")
print(df.isnull().sum())


# In[14]:


x = df.drop("quality", axis=1)
y = df["quality"]
normalizer = MinMaxScaler()
x_normalized = normalizer.fit_transform(x)
print("\nNormalized data(first 5 rows):")
print(pd.DataFrame(
    x_normalized, columns=x.columns
).head().round(3))


# In[16]:


standardizer = StandardScaler()
x_standardized = standardizer.fit_transform(x)
print("\nStandardized data(first 5 rows):")
print(pd.DataFrame(
    x_standardized, columns=x.columns
).head().round(3))


# In[19]:


pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_standardized)
print("\nPCA output (first 5 rows):")
print(pd.DataFrame(
    x_pca, columns=["PC1", "PC2"]
).head().round(3))


# In[20]:


print("\nOriginal number of features :", x.shape[1])
print("Reduced number of features:", x_pca.shape[1])


# In[22]:


print("\nVariance retained by each component:")
print(pca.explained_variance_ratio_.round(4))
print("Total variance retained:",
     round(sum(pca.explained_variance_ratio_)* 100, 2), "%")


# In[23]:


plt.figure(figsize=(8, 5))
plt.scatter(x_pca[:, 0], x_pca[:, 1],
           c=y, cmap="viridis", alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Wine Quality Dataset")
plt.colorbar(label= "Wine Quality")
plt.show()


# In[ ]:




