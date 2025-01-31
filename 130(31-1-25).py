#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[2]:


data1 =pd.read_csv("NewspaperData.csv")
data1


# In[3]:


data1.info()


# In[12]:


data1.isnull().sum()


# In[20]:


cols=data1.columns
colors=['White','Black']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar=True)


# In[14]:


data1.describe()


# In[15]:


data1.boxplot()


# In[16]:


fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})
sns.histplot(data1["Newspaper"], kde=True, ax=axes[1], color='green', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Newspaper")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show


# In[17]:


plt.scatter(data1["Newspaper"],data1["daily"])


# In[18]:


plt.scatter(data1["daily"],data1["sunday"])


# In[ ]:




