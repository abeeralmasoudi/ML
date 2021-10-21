#!/usr/bin/env python
# coding: utf-8

# In[25]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error,r2_score
import warnings
warnings.filterwarnings("ignore")
#read data

data = pd.read_csv('D:\\ML project\\Used cars in KSA.csv')
var = 'car_model_year'
datax = pd.concat([data['car_price'],data[var]],axis = 1)
data1=data[data['car_price']<4e4].reset_index(drop=True)
Q1 = data1.car_model_year.quantile(0.25)
Q3 = data1.car_model_year.quantile(0.75)
Q1,Q3
IQR = Q3 - Q1 
IQR
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
lower_limit , upper_limit
no_out =data1 [(data1.car_model_year>lower_limit) & (data1.car_model_year<upper_limit)]
no_out

Q1 = no_out.car_driven.quantile(0.25)
Q3 = no_out.car_driven.quantile(0.75)
Q1,Q3
IQR = Q3 - Q1 
IQR
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
lower_limit , upper_limit
newdata=no_out[(no_out.car_driven>lower_limit) & (no_out.car_driven<upper_limit)]
x = newdata.iloc[:,[0,1,2,3,4]].values
y = newdata.iloc[:,[5]].values
le1=LabelEncoder()
x[:,0] =le1.fit_transform(x[:,0])
le2=LabelEncoder()
x[:,1] =le1.fit_transform(x[:,1])
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])],remainder='passthrough')#this one is for count every value repatedly not only 3 uniq
x=ct.fit_transform(x)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x=sc.fit_transform(x)
from sklearn.model_selection import train_test_split
(x_train,x_test,y_train,y_test) = train_test_split(x,y,test_size = 0.3,random_state=0)
#set lasso model on train set
lasso_model = Lasso().fit(x_train,y_train)
#lasso regression model constant
lasso_model.intercept_
lasso = Lasso()
coefs = []
alphas = np.random.randint(0,1000,100)
for a in alphas:
   lasso.set_params(alpha = a)
   lasso.fit(x_train,y_train)
   coefs.append(lasso.coef_)
lasso_model.predict(x_train)[:5]


# In[3]:


#calculate mean square error
y_pred = lasso_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))from yellowbrick.regressor import ResidualsPlot


# In[10]:


#calculate r_square
r2_score(y_test, y_pred)


# In[27]:


#Model Tuning(using the LassoCV method to find the optimum alpha value).
lasso_cv_model = LassoCV(alphas = np.random.randint(0,1000,100), cv = 10, max_iter = 100000).fit(x_train,y_train)
lasso_cv_model.alpha_


# In[28]:


#setup the Corrected Lasso model with optimum alpha value
lasso_tuned = Lasso().set_params(alpha = lasso_cv_model.alpha_).fit(x_train,y_train)
y_pred_tuned = lasso_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred_tuned))


# In[23]:


r2_score(y_test, y_pred_tuned)


# In[ ]:




