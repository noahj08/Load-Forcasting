#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
train = pd.read_pickle('train.pickle')
test = pd.read_pickle('test.pickle')


# In[2]:


train


# In[3]:


import numpy as np
import matplotlib.pyplot as plt 


# In[4]:


import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


train_x = train[0]
train_y = train[1]
test_x = test[0]
test_y = test[1]


# In[22]:


lm = LinearRegression()


# In[23]:


lm.fit(train_x, train_y)


# In[25]:


predictions = lm.predict(test_x)


# In[30]:


plt.scatter(test_y,predictions)
plt.title("Actual vs. Predicted Demand (kW)")
plt.xlabel("Actual (kW)")
plt.ylabel("Predicted (kW)")
plt.show()


# In[29]:


metrics.r2_score(test_y, predictions)


# In[31]:


metrics.mean_squared_error(test_y, predictions)


# In[ ]:




