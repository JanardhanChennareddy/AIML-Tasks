#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf 
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[18]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[19]:


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

# In[20]:


cars.info()


# In[21]:


cars.isna().sum()


# In[22]:


cars.isna().diff()


# In[23]:


cars.isna().product()


# # Observations about info(),missing values
# .There are no missing values
# .there are 81 observations(81 different cars data)
# .the data types of the columns are also relvant and valid

# In[24]:


fig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[25]:


fig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[26]:


fig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[27]:


fig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[28]:


cars[cars.duplicated()]


# Pair plots and correlation and coefficients

# In[29]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[30]:


cars.corr()


# # Observations from correlation plots and coefficients
# ~b/w x nd y all the varaibles are showing moderate to high correlation strengths,highest being b/w HP and MPG
# 
# ~Therefore this dataset qualifies for building a multiple linear regression model to predict MPG
# 
# ~Among x columns (x1,x2,x3 and x4) some very high correlation strenghts are observed b/w SP vs HP, VOL vs  WT
# 
# ~The high correlation among x columns is not desirable as it might lead to multicollineary problem

# In[31]:


model1 = smf.ols('MPG~WT+VOL+HP+SP', data=cars).fit()


# In[32]:


model1.summary()


# # Observations from model summary
# ~The R-squared and adjusted R-squared values are good and about 75% of variability in Y is explained by X columns
# 
# ~The probability value with respect to F-statistic is close to zero, including that all or some of x columns are significant
# 
# !the p-valus for VOL and WT are higher than 5% indicating some interaction issue among themselves,which need to be further explored

# In[43]:


df1 =pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[44]:


df1 =pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.tail()


# In[45]:


df1 =pd.DataFrame()
df1["actual_y1"] = cars["VOL"]
df1.head()


# In[46]:


df1 =pd.DataFrame()
df1["actual_y1"] = cars["HP"]
df1.head()


# In[47]:


df1 =pd.DataFrame()
df1["actual_y1"] = cars["SP"]
df1.head()


# In[ ]:




