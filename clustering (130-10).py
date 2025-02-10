#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.cluster import KMeans


# In[2]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[3]:


Univ.info()


# In[4]:


Univ.describe()


# In[5]:


Univ.isnull()


# In[6]:


Univ1 = Univ.iloc[:,1:]


# In[7]:


Univ1


# In[8]:


Univ1.columns


# In[14]:


cols=Univ1.columns


# In[15]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols)
scaled_Univ_df


# In[ ]:




