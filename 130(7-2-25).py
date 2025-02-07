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

# In[4]:


cars.info()


# In[5]:


cars.isna().sum()


# In[6]:


cars.isna().diff()


# In[7]:


cars.isna().product()


# # Observations about info(),missing values
# .There are no missing values
# .there are 81 observations(81 different cars data)
# .the data types of the columns are also relvant and valid

# In[8]:


fig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[9]:


fig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[10]:


fig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[11]:


fig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[12]:


cars[cars.duplicated()]


# Pair plots and correlation and coefficients

# In[13]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[14]:


cars.corr()


# # Observations from correlation plots and coefficients
# ~b/w x nd y all the varaibles are showing moderate to high correlation strengths,highest being b/w HP and MPG
# 
# ~Therefore this dataset qualifies for building a multiple linear regression model to predict MPG
# 
# ~Among x columns (x1,x2,x3 and x4) some very high correlation strenghts are observed b/w SP vs HP, VOL vs  WT
# 
# ~The high correlation among x columns is not desirable as it might lead to multicollineary problem

# In[15]:


model1 = smf.ols('MPG~WT+VOL+HP+SP', data=cars).fit()


# In[16]:


model1.summary()


# # Observations from model summary
# ~The R-squared and adjusted R-squared values are good and about 75% of variability in Y is explained by X columns
# 
# ~The probability value with respect to F-statistic is close to zero, including that all or some of x columns are significant
# 
# !the p-valus for VOL and WT are higher than 5% indicating some interaction issue among themselves,which need to be further explored

# In[17]:


df1 =pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[18]:


df1 =pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.tail()


# In[19]:


df1 =pd.DataFrame()
df1["actual_y1"] = cars["VOL"]
df1.head()


# In[20]:


df1 =pd.DataFrame()
df1["actual_y1"] = cars["HP"]
df1.head()


# In[21]:


df1 =pd.DataFrame()
df1["actual_y1"] = cars["SP"]
df1.head()


# In[22]:


cars =pd.DataFrame (cars, columns=["HP", "VOL", "SP","WT", "MPG"])
cars.head()


# In[23]:


pred_y1 =model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[24]:


from sklearn.metrics import mean_squared_error
mse =mean_squared_error(df1["actual_y1"], df1["pred_y1"])
print("MSE: ",mse)
print("RMSE :",np.sqrt(mse))


# # checking for multicollinearity among x-columns using VIF method

# In[25]:


rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# # Observations for VIF values:
# ~The ideal range f VIF values shall be between 0 to 10.however slightly higher values can be tolerrated
# 
# ~as seen as from the very high VIF values for VOL

# In[26]:


cars1 = cars.drop("WT", axis=1)
cars.head()


# In[34]:


model2=smf.ols("MPG~ HP+VOL+SP", data=cars1).fit()
model2.summary()


# In[40]:


df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[41]:


pred_y2 = model2.predict(cars1.iloc[:, 0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[42]:


from sklearn.metrics import mean_squared_error
mse =mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE: ",mse)
print("RMSE :",np.sqrt(mse))


# # Observations from model2 summary()
# ~The adjusted R-squared value improved slightly to 0.76
# 
# ~ All the p-values for model parametres are less than 5% hence they are significant 
# 
# ~Therefore the HP,VOL,SP columns are finalised
# 
# ~THere is no improvemnet in MSE value

# In[ ]:




