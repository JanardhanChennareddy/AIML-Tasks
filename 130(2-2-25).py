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

# In[15]:


x=data1["daily"]
y=data1["sunday"]
plt.scatter(data1["daily"],data1["sunday"])
plt.xlim(0, max(x) +100)
plt.ylim(0,max(y)+100)
plt.show()


# In[16]:


data1["daily"].corr(data1["sunday"])


# In[17]:


data1[["daily" , "sunday"]].corr()


# # Fit a linear regression model

# In[18]:


import statsmodels.formula.api as smf
model1 =smf.ols("sunday~daily",data=data1).fit()


# In[19]:


model1.summary()


# # Interpretation
# 

# In[20]:


x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x,y, color ="pink", marker = "s", s =50)
b1 =1.43


# In[21]:


model1.params


# In[22]:


print(f'model t-values:\n{model1.tvalues}\n-------------\nmodel p-values:\n{model1.pvalues}')


# In[23]:


model1.rsquared,model1.rsquared


# In[24]:


newdata=pd.Series([200,300,1500])


# In[25]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[26]:


model1.predict(data_pred)


# In[28]:


pred= model1.predict(data1["daily"])
pred


# In[29]:


data1["Y_hat"] = pred
data1


# In[34]:


data1["residuals"]= data1["sunday"]-data1["Y_hat"]
data1


# In[36]:


mse= np.mean((data1["daily"]-data1["Y_hat"])**2)
rmse=np.sqrt(mse)
print("MSE: ",mse)
print("RMSE: ",rmse)


# In[ ]:




