'''Flood prediction in rivers using Machine learning( Disaster Management)'''


#Model-Logistic Regression
#Class-skewed class

#importing required libraries
import pandas as pd
from datetime import datetime
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer,MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
import numpy as np
import seaborn as sns
import plotly.graph_objs as go
import chart_studio.plotly as py
import plotly
from sklearn.externals import joblib

#Our data set
filenames=['Cauvery','Godavari','Krishna','Mahanadi','Son']

Total_pred=pd.DataFrame()

for filename in filenames:
    
    #Extracting files
    data1=pd.read_excel('data/'+filename+'.xlsx')
    
    #Filling mean value in vacant spaces
    for i in range(1,len(data1.columns)):
	    data1[data1.columns[i]] = data1[data1.columns[i]].fillna(data1[data1.columns[i]].mean())
    
    #Creating the target
    y=data1['Flood']
    
    #App the result to 0 or 1
    for i in range(len(y)):
        if(y[i] >= 0.1):
            y[i]=1
            
    #Preprocessing data - date/month/year feature for train/test data split
    d1=pd.DataFrame()
    d1["Day"]=data1['Date']
    d1['Months']=data1['Date']
    d1['Year']=data1['Date']
    data1['Date']=pd.to_datetime(data1['Date'])
    d1["Year"]=data1.Date.dt.year
    d1["Months"]=data1.Date.dt.month
    d1["Day"]=data1.Date.dt.day
    
    #Drop the target column from features
    data1.drop('Flood',axis=1,inplace=True)
    
    #Drop the date column because we have the preprocessed form
    data1.drop('Date',inplace=True,axis=1)
    data1=pd.concat([d1,data1],axis=1)
    
    #for good result split train test with equal priority
    locate=0
    for i in range(len(data1["Day"])):
        if(data1["Day"][i]==31 and data1["Months"][i]==12 and data1["Year"][i]==2015):
            locate=i
            break
    i=locate+1
    
    #Train/Test data split
    x_train=data1.iloc[0:i,:]
    y_train=y.iloc[0:i]
    x_test=data1.iloc[i:,:]
    y_test=y.iloc[i:]
    
    #Drop day/month/year features because it is unnecessary
    x_train.drop(labels=['Day','Months','Year'],inplace=True,axis=1)
    x_test.drop(labels=['Day','Months','Year'],inplace=True,axis=1)
    
    #Nearmiss Algo for getting good result in skewed class
    from imblearn.under_sampling import NearMiss 
    nr = NearMiss() 
    
    X_train_res, Y_train_res = nr.fit_sample(x_train, y_train)
    x_train, y_train = shuffle( X_train_res, Y_train_res, random_state=0)
    
    #Logistic regression- binary classification
    
    from sklearn.linear_model import LogisticRegression 
    lr = LogisticRegression() 
    
    #Fit the model
    lr.fit(x_train, y_train.ravel())
    
    #Predict the result
    y_predict=lr.predict(x_test)
    
    #Accuracy in train/test datasets
    print(lr.score(x_train,y_train))
    print(lr.score(x_test,y_test))
    
    #Precision/recall result skewed class requires high recall value
    print(classification_report(y_test, y_predict))
    
    #Mean absolute error
    mae=mean_absolute_error(y_test, y_predict)
    print("mean_absolute_error=",mae)
    
    #Confusion matrix
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test,y_predict))
    
    Total_pred[filename]=y_predict
            
print(Total_pred)
Total_pred.to_csv('Output/Output.csv')
