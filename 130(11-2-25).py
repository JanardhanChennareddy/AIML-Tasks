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


Univ1 = Univ.iloc[:,1:]


# In[6]:


Univ1


# In[7]:


Univ1.columns


# In[8]:


cols=Univ1.columns


# In[9]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols)
scaled_Univ_df


# In[10]:


from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[11]:


clusters_new.labels_


# In[12]:


set(clusters_new.labels_)


# In[13]:


Univ['clusterid_new'] = clusters_new.labels_


# In[14]:


Univ


# In[15]:


Univ.sort_values(by ='clusterid_new' )


# In[16]:


Univ.iloc[:,1:].groupby("clusterid_new").mean()


# # Observations:
# ~Cluster2 appears to be the top rated universities cluster as the cut off score, top10,SFratio parameter mean values are highest
# 
# ~Cluster1 appears to occupy the middle level rated universities
# 
# ~Cluster0 comes as the lower level rated universities

# In[20]:


Univ[Univ['clusterid_new']==0]


# In[28]:


wcss = []
for i in range(1, 20):
    
    kmeans =  KMeans(n_clusters=i,random_state=0 )
    kmeans.fit(scaled_Univ_df)
    
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('No of clusters')
plt.show()


# # Observations:
# from the above graph we can observe k=3 or 4 which indicates the elbow just joint i.e rate of change of slope decreases

# In[30]:


from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[ ]:




