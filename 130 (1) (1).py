#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[28]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[29]:


data.info()


# In[30]:


print(type(data))
print(data.shape)
print(data.size)


# In[31]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[32]:


data1.info()


# In[33]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[34]:


data[data1.duplicated(keep = False)]


# In[35]:


data1.drop_duplicates(keep='first', inplace = True)
data1


# In[36]:


data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# impute the missing values

# In[37]:


data1.info()


# In[38]:


data1.isnull().sum()


# In[39]:


cols = data1.columns
colors = ['black','red']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[40]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[41]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[42]:


data1['Solar'] = data1['Solar'].fillna(mean_ozone)
data1.isnull().sum()


# In[43]:


print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[44]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[45]:


print(data1["Month"].value_counts())
mode_month = data1["Month"].mode()[0]
print(mode_month)


# In[46]:


data1["Month"] = data1["Month"].fillna(mode_weather)
data1.isnull().sum()


# # 

# # Detection of outliers in the columns

# In[47]:


fig, axes = plt.subplots(2, 1, figsize=(8,6), gridspec_kw={'height_ratios': [1,3]})

sns.boxplot(data=data1["Ozone"], ax=axes[0], color='red',width=0.5,orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")

sns.histplot(data1["Ozone"],kde=True, ax=axes[1], color='lightgreen',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_ylabel("Ozone Levels")
axes[1].set_xlabel("Frequency")
plt.tight_layout()
plt.show()


# Observations

# In[50]:


plt.figure(figsize=(6,2))
plt.boxplot(data1["Ozone"], vert=False)


# In[52]:


plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert=False)
[item.get_xdata() for item in boxplot_data['fliers']]


# In[53]:


data1["Ozone"].describe()


# In[58]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]
for x in data1["Ozone"]:
    if ((x < (mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)


# In[59]:


import scipy.stats as stats

plt.figure(figsize=(8, 6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# In[61]:


sns.violinplot(data=data1["Ozone"], color='red')
plt.title("violin Plot")


# In[ ]:




