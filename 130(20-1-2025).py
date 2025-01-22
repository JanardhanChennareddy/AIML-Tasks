#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


df = pd.read_csv("universities.csv")
df


# In[5]:


df.sort_values(by="GradRate",ascending=True)


# In[6]:


df.sort_values(by="GradRate",ascending=False)


# In[7]:


df[df["GradRate"]>=95]


# In[8]:


df[(df["GradRate"]>=80) & (df["SFRatio"]<=12)]


# In[9]:


sal = pd.read_csv("Salaries.csv")
sal


# In[10]:


sal[["salary"]].groupby(sal["rank"]).mean()


# In[11]:


sal[["salary","phd","service"]].groupby(sal["rank"]).mean()


# In[12]:


sal[["phd"]].groupby(sal["rank"]).mean()


# In[13]:


df = pd.read_csv("universities.csv")
df


# In[14]:


#mean value of SAT score
np.mean(df["SAT"])


# In[15]:


#median value of SAT score
np.median(df["SAT"])


# In[16]:


#variance
np.var(df["SFRatio"])


# In[17]:


df.describe()


# In[ ]:




