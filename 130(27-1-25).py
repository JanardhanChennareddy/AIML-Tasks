#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[29]:


data1 =pd.read_csv("data_clean.csv")
print(data1)


# In[30]:


data1.info()


# In[31]:


print(type(data1))
print(data1.shape)
print(data1.size)


# In[32]:


data1.info()


# In[33]:


print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[36]:


fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios':[1, 3]})
sns.boxplot(data=data1["Ozone"], ax=axes[0], color='black', width=0.5, orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone levels")

sns.histplot(data1["Ozone"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone levels")
axes[1].set_ylabel("Frequency")

plt.tight_layout()

plt.show()



# In[ ]:




