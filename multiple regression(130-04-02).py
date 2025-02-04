#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf 
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[6]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[9]:


cars = pd.read_csv("Cars.csv")
cars.tail()


# In[11]:


cars=pd.DataFrame(cars, columns=["HP", "VOL", "SP", "WT", "MPG"])
cars.head()


# # descriptions of columns
# MPG : mileage of the car (mile per galion)
# Hp : horse power of the car
# VOL: volume of the car 
# SP:top speed of the car 
# WT:weight of the car

# In[ ]:




