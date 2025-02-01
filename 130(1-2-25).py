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


# In[4]:


data1.isnull().sum()


# In[5]:


cols=data1.columns
colors=['White','Black']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar=True)


# In[6]:


data1.describe()


# In[7]:


data1.boxplot()


# In[8]:


fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})
sns.histplot(data1["Newspaper"], kde=True, ax=axes[1], color='green', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Newspaper")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show


# In[9]:


plt.scatter(data1["Newspaper"],data1["daily"])


# In[10]:


plt.scatter(data1["daily"],data1["sunday"])


# In[11]:


data1.info()


# In[12]:


data1.isnull().sum()


# In[13]:


data1.describe()


# In[14]:


plt.figure(figsize=(6,3))
plt.title("Boxplot for Daily Sales")
plt.boxplot(data1["daily"], vert= False)
plt.show()


# # scatter plot and co relation strength

# In[19]:


x=data1["daily"]
y=data1["sunday"]
plt.scatter(data1["daily"],data1["sunday"])
plt.xlim(0, max(x) +100)
plt.ylim(0,max(y)+100)
plt.show()


# In[20]:


data1["daily"].corr(data1["sunday"])


# In[23]:


data1[["daily" , "sunday"]].corr()


# # Fit a linear regression model

# In[27]:


import statsmodels.formula.api as smf
model1 =smf.ols("sunday~daily",data=data1).fit()


# In[28]:


model1.summary()


# In[ ]:




