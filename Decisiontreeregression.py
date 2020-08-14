#!/usr/bin/env python
# coding: utf-8

# In[1]:


#building a decision tree regression model to predict the workers salary relating to their year of working experience
#importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#importing the dataset with pandas
salaries=pd.read_excel("C:/Users/NifemiDev/Downloads/Position_Salaries.xlsx")


# In[3]:


salaries.head()


# In[4]:


#converting the dependent and independent values to matrix
x=salaries.iloc[:,1].values
y=salaries.iloc[:,2].values


# In[6]:


#fitting the decision treeregression to the dataset
from sklearn.tree import DecisionTreeRegressor


# In[7]:


regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x.reshape(-1,1),y)


# In[9]:


#predicting the result
y_pred=regressor.predict(np.array([6.5]).reshape(-1,1))


# In[10]:


y_pred


# In[12]:


#visualize the decision tree
x_grid=np.arange(min(x.reshape(-1,1)),max(x.reshape(-1,1)),0.01)
x_grid=x_grid.reshape((len(x_grid),1))


# In[13]:


plt.scatter(x.reshape(-1,1),y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.show()


# In[ ]:




