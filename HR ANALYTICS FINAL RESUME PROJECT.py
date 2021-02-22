#!/usr/bin/env python
# coding: utf-8

# Based on the Dataset given we have to predict using the ml algorithm who will get promoted and who will not get promoted
# 1 is promoted and 0 is nt promted

#  lets first load all libraries:

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns


# In[6]:


data=pd.read_csv('HR_train.csv') ##reading the dataset


# In[7]:


data.head() ##to print first 5 records


# In[8]:


data.shape ##to know the no of records


# In[9]:


data.columns


# In[10]:


len(data)


# In[11]:


data.info()


# In[12]:


data.describe()


# In[13]:


data.isnull().sum()


# In[ ]:


##education and previous year rating has missing values


# In[14]:


data.dtypes


# In[15]:


data['is_promoted'].value_counts()


# In[16]:


data['KPIs_met >80%'].value_counts()


# In[17]:


data['previous_year_rating'].value_counts()


# In[18]:


data['gender'].value_counts()


# In[19]:


sns.countplot(x='is_promoted',hue='gender',data=data)


# In[20]:


sns.countplot(x='previous_year_rating',hue='gender',data=data)


# In[21]:


data.head(2)


# In[22]:


sns.countplot(x='education',hue='gender',data=data)


# In[24]:


data.head(2)


# In[25]:


data1=pd.read_csv("HR_test.csv")


# In[26]:


data1.head()


# In[27]:


data1.shape


# In[28]:


data.shape


# In[29]:


data1.isnull().sum()


# In[ ]:


##education and previous_year_rating values are missing


# In[30]:


data1.describe()


# In[31]:


import missingno as msno
msno.matrix(data1)


# since dataset is not balanced,we will balanced it and impute the missing values 

# In[32]:


data.head(2)


# In[35]:


data['education'].value_counts()


# In[36]:


data.isnull().sum()


# In[38]:


type('education')


# In[40]:


data['education'].fillna(data['education'].mode()[0],inplace=True)


# In[41]:


data.isnull().sum()


# In[42]:


type('previous_year_rating')


# In[45]:


data['previous_year_rating'].value_counts()


# In[43]:


data['previous_year_rating'].fillna(1,inplace=True)


# In[44]:


data.isnull().sum()


# In[46]:


data1.head(2)


# In[47]:


data1['education'].fillna(data1['education'].mode()[0],inplace=True)


# In[48]:


data1.isnull().sum()


# In[49]:


data1['previous_year_rating'].fillna(1,inplace=True)


# In[50]:


data1.isnull().sum()


# In[51]:


data=data.drop('employee_id',axis=1)


# In[52]:


data.head(2)


# In[129]:


data[data['KPIs_met >80%']==1]


# In[131]:


data[data['avg_training_score']<50] 


# In[130]:


data[data['avg_training_score']>80] 


# In[127]:


test=data1.drop('employee_id',axis=1)


# In[55]:


x_test=test


# In[56]:


test.columns


# In[57]:


test.head(2)


# In[58]:


x_test=pd.get_dummies(x_test)


# In[59]:


x_test.head()


# In[60]:


x_test.columns


# In[ ]:


##splitting the data 


# In[61]:


data.head(2)


# In[62]:


test.head(2)


# In[63]:


x=data.iloc[:,:-1]


# In[67]:


y=data.iloc[:,-1]


# In[68]:


print("shape of x:",x.shape)


# In[69]:


print("shape of y:",y.shape)


# In[70]:


x=pd.get_dummies(x)


# In[71]:


x.head(2)


# In[72]:


x.columns


# In[73]:


x.head(5)


# In[ ]:


##data is imbalanced


# In[74]:


from imblearn.over_sampling import SMOTE


# In[75]:


x_sample,y_sample=SMOTE().fit_sample(x,y.values.ravel())


# In[76]:


x_sample=pd.DataFrame(x_sample)


# In[77]:


y_sample=pd.DataFrame(y_sample)


# In[82]:


print(x_sample.shape)


# In[79]:


print(y_sample.shape)


# In[80]:


from sklearn.model_selection import train_test_split


# In[83]:


x_train,x_valid,y_train,y_valid=train_test_split(x_sample,y_sample,test_size=0.2,random_state=0)


# In[84]:


print("shape of x_train:",x_train.shape)


# In[85]:


print("shape of y_train:",y_train.shape)


# In[137]:


x_train


# In[138]:


from sklearn.preprocessing import StandardScaler


# In[139]:


sc=StandardScaler()


# In[140]:


x_train=sc.fit_transform(x_train)


# In[141]:


x_test=sc.fit_transform(x_test)


# In[142]:


x_valid=sc.transform(x_valid)


# In[143]:


from sklearn.ensemble import RandomForestClassifier


# In[144]:


random_forest=RandomForestClassifier()


# In[145]:


random_forest.fit(x_train,y_train)


# In[146]:


y_pred=random_forest.predict(x_test)


# In[147]:


from sklearn import metrics


# In[99]:


cnf_matrix = metrics.confusion_matrix(y_test,y_pred)


# In[100]:


print(cnf_matrix)


# In[101]:


print("Accuracy:",metrics.accuracy_score(y_test,y_pred))


# In[123]:


from sklearn.metrics import classification_report


# In[124]:


cf=classification_report(y_pred,y_test)


# In[125]:


print(cf)


# In[102]:


from sklearn.tree import DecisionTreeClassifier


# In[103]:


model= DecisionTreeClassifier() 


# In[104]:


model.fit(x_train,y_train) 


# In[110]:


y_predictions= model.predict(x_test)


# In[111]:


from sklearn.metrics import accuracy_score,classification_report


# In[112]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix)


# In[113]:


from sklearn import metrics


# In[114]:


print("Accuracy:",metrics.accuracy_score(y_test,y_predictions))


# In[115]:


from sklearn.ensemble import RandomForestClassifier


# In[116]:


from sklearn.model_selection import cross_val_score


# In[117]:


randomforest_classifier= RandomForestClassifier(n_estimators=10)


# In[118]:


score=cross_val_score(randomforest_classifier,x,y,cv=10)


# In[119]:


score.mean()


# In[120]:


from sklearn.metrics import classification_report


# In[121]:


cf=classification_report(y_pred,y_test)


# In[122]:


print(cf)

