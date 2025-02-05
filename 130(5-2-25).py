#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf 
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


cars = pd.read_csv("Cars.csv")
cars.tail()


# # descriptions of columns
# MPG : mileage of the car (mile per galion)
# Hp : horse power of the car
# VOL: volume of the car 
# SP:top speed of the car 
# WT:weight of the car

# Assumptions in Multilinear regression
# 1.Linearity: The realationship b/w the pedictors(X) and the response(Y) is linear.
# 2.Independence:observations are independent of each other 
# 3.Homoscedasticity:the residuals(Y-Y_hat) EXHIBIT CONSTANT VARIANCE AT ALL LEVELS OF THE PREDICTOR
# 4.nORMAL DISTRIBUTION OF ERRORS:The indepedent variables  should not be too highly correalated with each other.
# 5.No mutlicollinearity:the independent variables should not be to highly correalted with each other.
# Violations of these assumptions may lead tio inefficiency in thye regession parameter and reliable predictions.
# 

# # EDA

# In[5]:


cars.info()


# In[6]:


cars.isna().sum()


# In[8]:


cars.isna().diff()


# In[9]:


cars.isna().product()


# # Observations about info(),missing values
# .There are no missing values
# .there are 81 observations(81 different cars data)
# .the data types of the columns are also relvant and valid

# In[14]:


fig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[18]:


fig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[19]:


fig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[20]:


fig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[21]:


cars[cars.duplicated()]


# Pair plots and correlation and coefficients

# In[22]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[23]:


cars.corr()


# In[ ]:




