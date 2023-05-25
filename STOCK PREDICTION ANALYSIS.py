#!/usr/bin/env python
# coding: utf-8
Project:Stock Price Prediction
Compiler:Mohammad Anas Ansari
# In[2]:


#install Quandl
get_ipython().system('pip install quandl')


# In[3]:


#IMPORTING NECESSARY LIBRARY
import quandl as qdl
import pandas as pd 
import numpy as np


# In[5]:


#GETTING KEY
qdl.ApiConfig.api_key = "" 


# In[66]:


#SELECTING STOCK FROM DEFINED DATES
stock=qdl.get('NSE/PNB',start_date='2018-01-01',end_date='2018-01-31')


# In[7]:


print(stock)


# In[8]:


#LIBRARY FOR VISUALIZATION
import seaborn as sns
import matplotlib.pyplot as plt


# In[9]:


#plot to show the closing price of stock
plt.plot(stock.index, stock['Close'])
plt.title('PNB Stock Price')
plt.ylabel('Price (₹)');
plt.show()  
plt.figure(figsize=(2000,10))
           
 


# # Now Converting into DataFrame for Further Analysis

# In[10]:


df=pd.DataFrame(stock)


# In[11]:


df


# In[12]:


#Converting Into CSV file
df.to_csv("PNB_csv")


# In[13]:


data=pd.read_csv("PNB_csv")


# In[14]:


data


# In[67]:


#INFO OF DATAFRAME
df.info()

Checking the Null Values
# In[68]:


df.describe()


# In[15]:


#CHECKING THE NULL VALUES
data.isnull()


# In[16]:


data.isnull().sum()


# There is no null values in this dataset.

# In[17]:


#CORRELATION
data.corr()


# In[18]:


#PLOTTING CORRELATION
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,cmap="crest",center=0)
plt.show()

1)Open is a DEPENDENT VARIABLE
2)while the rest are INDEPENDENT VARIABLE
3)DATE coloumn Not required For the analysis
# In[19]:


#Date Coloumn is to be removed from the dataset 
x=data.loc[:,"High":"Turnover (Lacs)"].values


# In[20]:


data.loc[:,"High":"Turnover (Lacs)"]


# In[21]:


x


# In[22]:


#Dependent variable
y=data.loc[:,"Open"].values


# In[23]:


data.loc[:,"Open"]


# In[24]:


y


# In[25]:


type(x),type(y)


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[28]:


x_train


# In[29]:


x_test


# In[30]:


y_test


# In[31]:


y_train


# # FITTING THE MODEL LINEAR REGRESSION

# In[57]:


from sklearn.linear_model import LinearRegression


# In[58]:


LR=LinearRegression()


# In[59]:


LR.fit(x_train,y_train)


# In[60]:


LR.score(x_test,y_test)


# In[61]:


Test_Data=[[49.41,48.56,49.66,50.60,50.12,50.0]]


# In[62]:


Test_Data


# In[63]:


type(Test_Data)


# In[64]:


Predicted_Price=LR.predict(Test_Data)


# In[65]:


print(Predicted_Price)


# The predicted Price is 51.299 while the real price on that day is 50.55.
# 
# CONCLUSION: Linear Regression Model is working fine On this Dataset 
