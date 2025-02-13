#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mlxtend')


# In[2]:


import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[3]:


titanic = pd.read_csv("Titanic.csv")
titanic


# In[4]:


titanic.info()


# In[5]:


titanic.describe()


# In[6]:


titanic.isnull().sum()


# In[7]:


titanic.columns


# # Observations:
# 1.The rows of the titanic are rows,class,age,survived
# 
# 2.no null data values
# 
# 3.all the  columns has same datatype object

# In[8]:


titanic['Class'].value_counts()


# In[9]:


titanic['Age'].value_counts()


# In[10]:


titanic['Survived'].value_counts()


# In[11]:


counts = titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# # Observations:
# 1.The maximum members are the crew
# 2.The second highest are 3rd class 
# 3.The middest value is 1st class
# 4.The lowest value is 2nd class

# In[12]:


counts = titanic['Age'].value_counts()
plt.bar(counts.index, counts.values)


# # Observations:
# 1.Adults are nmore than childrens

# In[13]:


counts = titanic['Survived'].value_counts()
plt.bar(counts.index, counts.values)


# # Observaton:
# 1.no has highest survived capacity

# In[14]:


counts = titanic['Gender'].value_counts()
plt.bar(counts.index, counts.values)


# # Observations:
# 1. males are the higher than females

# In[15]:


df=pd.get_dummies(titanic, dtype=int)
df.head()


# In[16]:


df.info()


# # Apriori Algorithm

# In[17]:


frequent_itemsets = apriori(df, min_support = 0.05,use_colnames=True,max_len=None)
frequent_itemsets


# In[18]:


frequent_itemsets.iloc[62,1]


# In[19]:


frequent_itemsets.info()


# In[20]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules


# In[21]:


rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=1.0)
rules


# In[22]:


rules.sort_values(by='lift', ascending=True)


# In[24]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift']].hist(figsize=(15,7))
plt.show()


# In[ ]:




