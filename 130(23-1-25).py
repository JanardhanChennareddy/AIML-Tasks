#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np


# In[35]:


df = pd.read_csv("universities.csv")
df


# In[36]:


df.sort_values(by="GradRate",ascending=True)


# In[37]:


df.sort_values(by="GradRate",ascending=False)


# In[38]:


df[df["GradRate"]>=95]


# In[39]:


df[(df["GradRate"]>=80) & (df["SFRatio"]<=12)]


# In[40]:


sal = pd.read_csv("Salaries.csv")
sal


# In[41]:


sal[["salary"]].groupby(sal["rank"]).mean()


# In[42]:


sal[["salary","phd","service"]].groupby(sal["rank"]).mean()


# In[43]:


sal[["phd"]].groupby(sal["rank"]).mean()


# In[44]:


df = pd.read_csv("universities.csv")
df


# In[45]:


#mean value of SAT score
np.mean(df["SAT"])


# In[46]:


#median value of SAT score
np.median(df["SAT"])


# In[47]:


#variance
np.var(df["SFRatio"])


# In[48]:


df.describe()


# In[49]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[50]:


plt.figure(figsize=(6,3))
plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# In[51]:


s= [20,15,10,25,30,35,28,40,45,60]
scores = pd.Series(s)
scores


# In[52]:


plt.boxplot(scores, vert=False)


# In[53]:


plt.boxplot(scores, vert=True)


# In[54]:


s= [20,15,10,25,30,35,28,40,45,60,120,150]
scores = pd.Series(s)
scores


# In[55]:


plt.boxplot(scores, vert=False)


# In[56]:


plt.boxplot(scores, vert=True)


# # identify outliers in universities dataset

# In[57]:


df = pd.read_csv("universities.csv")
df


# In[59]:


plt.boxplot(df["GradRate"], vert=False)


# In[60]:


plt.boxplot(df["SAT"], vert=False)


# In[61]:


plt.boxplot(df["Accept"], vert=False)


# In[63]:


plt.boxplot(df["Top10"], vert=False)


# In[ ]:




