#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""This program predicts corona virus infection by using machine learning model."""
#Importing necessary dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


# In[2]:


#importing Csv file and store the data into dataFrame "df".
df=pd.read_csv("infected_countries.csv")
# print(df.head())


# In[3]:


#Getting necessary inputs from the user
country_name=input("Enter Country Name:")
forecast_out=int(input("Enter no of days of forecast:"))
train_percent=float(input("Enter % of data you want to train:"))


# In[4]:


#getting dataframe for the country
df1=df[[country_name]]
print(df1)
#create another column (the target for dependent variable) shifted "n" days up
df1["Prediction"]=df1[[country_name]].shift(-forecast_out)
#print new data set
# print(df1.tail())


# In[5]:


#create the independent data set (X) 
#convert data frame into numpyarray
X=np.array(df1.drop(["Prediction"],1))
#remove last "n" rows
X=X[:-forecast_out]
print(X)


# In[6]:


#create the dependent dataset (y)
#convert dataframe to an np array (all of the values including NaN's)
y=np.array(df1["Prediction"])
#Get all of the y values except the last "n" rows
y=y[:-forecast_out]
print(y)


# In[7]:


#split data into x% training and (100-x)% testing
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=1-train_percent/100)


# In[8]:


#create and train our model using support vector machine (Regressor)
svr_rbf=SVR(kernel="rbf",C=1e3,gamma=0.1)
svr_rbf.fit(x_train,y_train)


# In[9]:


#testing model: score returns the cofficient of determination of R^2 of the prediction.
#The best possible score is 1.0
svm_confidence=svr_rbf.score(x_test,y_test)
print("svm confidence:",svm_confidence)


# In[10]:


#create and train the Linear Regression Model
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[11]:


#testing model: score returns the cofficient of determination of R^2 of the prediction.
#best possible value is 1.
lr_confidence=lr.score(x_test,y_test)
print("lr confidence:",lr_confidence)


# In[12]:


#set x_forecast equal to the last 30 rows of the original data set from the column
x_forecast=np.array(df1.drop(["Prediction"],1))[-forecast_out:]
print(x_forecast)


# In[13]:


#print the linear regression model predictions for the next "n" days
lr_prediction=lr.predict(x_forecast)
print(lr_prediction)


# In[14]:


#print the support vector regression model predictions for the next "n" days
svm_prediction=svr_rbf.predict(x_forecast)
print(svm_prediction)


# In[15]:


#Appending predicted outcomes of linear regression model into original column
df_lronly=pd.DataFrame(lr_prediction,columns=[country_name])
df_lr=pd.concat([df1,df_lronly],ignore_index=True)
print(df_lr)


# In[16]:


#Appending predicted outcomes of vector regression model into original column
df_svmonly=pd.DataFrame(svm_prediction,columns=[country_name])
df_svm=pd.concat([df1,df_svmonly],ignore_index=True)
print(df_svm)


# In[17]:


#plotting the data:
style.use("classic")
plt.plot(df_lr[country_name],"r--",label="Predicted using LR model")
plt.plot(df_svm[country_name],"g--",label="Predicted using SVM model.")
plt.plot(df1[country_name],label="Original Data")
plt.title(f"Corona virus prediction for {country_name}")
plt.xlabel("Days")
plt.ylabel("No of Confirmed Cases")
plt.legend()
plt.show()


# In[ ]:




