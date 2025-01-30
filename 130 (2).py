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


data[data1.duplicated(keep = False)]


# In[9]:


data1.drop_duplicates(keep='first', inplace = True)
data1


# In[10]:


data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# impute the missing values

# In[11]:


data1.info()


# In[12]:


data1.isnull().sum()


# In[13]:


cols = data1.columns
colors = ['black','red']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[14]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[15]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[16]:


data1['Solar'] = data1['Solar'].fillna(mean_ozone)
data1.isnull().sum()


# In[17]:


print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[18]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[19]:


print(data1["Month"].value_counts())
mode_month = data1["Month"].mode()[0]
print(mode_month)


# In[20]:


data1["Month"] = data1["Month"].fillna(mode_weather)
data1.isnull().sum()


# # 

# # Detection of outliers in the columns

# In[21]:


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

# In[22]:


plt.figure(figsize=(6,2))
plt.boxplot(data1["Ozone"], vert=False)


# In[23]:


plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert=False)
[item.get_xdata() for item in boxplot_data['fliers']]


# In[24]:


data1["Ozone"].describe()


# In[25]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]
for x in data1["Ozone"]:
    if ((x < (mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)


# In[26]:


import scipy.stats as stats

plt.figure(figsize=(8, 6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# In[27]:


sns.violinplot(data=data1["Ozone"], color='blue')
plt.title("violin Plot")
plt.show()


# In[28]:


sns.swarmplot(data=data1, x = "Weather",y = "Ozone",color="orange",palette="Set2",size=6)


# In[29]:


sns.kdeplot(data=data1["Ozone"], fill=True, color="green")
sns.rugplot(data=data1["Ozone"], color="black")


# In[30]:


sns.kdeplot(data=data1["Ozone"], fill=False, color="green")
sns.rugplot(data=data1["Ozone"], color="black")


# In[31]:


sns.boxplot(data = data1, x = "Ozone",y = "Weather")


# In[32]:


sns.boxplot(data = data1, x = "Ozone")


# In[33]:


sns.scatterplot(data["Wind"], data1["Temp"])
plt.show()


# In[34]:


plt.scatter(data1["Wind"], data1["Temp"])


# In[35]:


data1["Wind"].corr(data1["Temp"])


# In[36]:


data1["Temp"].corr(data1["Wind"])


# In[37]:


data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[38]:


data1.head()


# In[39]:


data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[40]:


data1_numeric.corr()Ozone	Solar	Wind	Temp
Ozone	1.000000	0.281113	-0.523738	0.597087
Solar	0.281113	1.000000	-0.034285	0.282027
Wind	-0.523738	-0.034285	1.000000	-0.441228
Temp	0.597087	0.282027	-0.441228	1.000000


# In[45]:


sns.pairplot(data1_numeric)


# In[47]:


data2=pd.get_dummies(data1,columns=['Month','Weather'])
data2


# In[49]:


data2=pd.get_dummies(data1,columns=['Weather'])
data2


# In[53]:


data1_numeric.values


# In[55]:


from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler

array = data1_numeric.values

scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(array)

set_printoptions(precision=2)
print(rescaledX[0:10,:])


# In[ ]:




