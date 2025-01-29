#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[3]:


data.info()


# In[4]:


print(type(data))
print(data.shape)
print(data.size)


# In[5]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[6]:


data1.info()


# In[7]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[8]:


data1[data1.duplicated(keep = False)]


# In[9]:


data1.drop_duplicates(keep='first', inplace = True)
data1


# In[10]:


data1.info()


# In[11]:


data1.isnull().sum()


# In[12]:


cols = data1.columns
colors = ['yellow','purple']
sns.heatmap(data[cols].isnull(),cmap=sns.color_palette(colors),cbar=True)


# In[13]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ",median_ozone)
print("Mean of Ozone: ",mean_ozone)


# In[14]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[15]:


data1['Ozone'] = data1['Ozone'].fillna(mean_ozone)
data1.isnull().sum()


# In[16]:


data1['Solar.R'] = data1['Solar.R'].fillna(mean_ozone)
data1.isnull().sum()


# In[17]:


print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[18]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[19]:


fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})
sns.boxplot(data=data1["Ozone"], ax=axes[0], color='skyblue', width=0.5, orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_title("Ozone Levels")
sns.histplot(data1["Ozone"], kde=True, ax=axes[1], color='yellow', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show


# In[20]:


plt.figure(figsize=(6,2))
plt.boxplot(data1["Ozone"], vert=False)


# In[21]:


plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert=False)
[item.get_xdata() for item in boxplot_data['fliers']]


# In[22]:


data1["Ozone"].describe()


# In[23]:


mu =data1["Ozone"].describe()[1]
sigma =data1["Ozone"].describe()[2]


for x in data1["Ozone"]:
    if ((x < (mu - 3*sigma)) or (x > (mu +  3*sigma))):
        print(x)


# In[25]:


import scipy.stats as stats
plt.figure(figsize=(8, 6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# In[26]:


sns.violinplot(data=data1["Ozone"], color='red')
plt.title("violin Plot")
plt.show()


# In[31]:


sns.swarmplot(data=data1, x = "Weather",y = "Ozone",color="black",palette="Set2",size=6)


# In[30]:


sns.kdeplot(data=data1["Ozone"], fill=True, color="pink")
sns.rugplot(data=data1["Ozone"], color="black")


# In[36]:


sns.boxplot(data =data1, x="Ozone", y="Weather")


# In[37]:


sns.boxplot(data =data1, x="Ozone")


# In[38]:


plt.scatter(data1["Wind"], data1["Temp"])


# In[39]:


data1["Wind"].corr(data1["Temp"])


# In[40]:


data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[ ]:




