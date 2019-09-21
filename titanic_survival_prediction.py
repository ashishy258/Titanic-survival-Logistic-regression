#!/usr/bin/env python
# coding: utf-8

# # libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import os
os.chdir("D:\\titanic")


# # importing data

# In[2]:


df=pd.read_csv("train.csv")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.shape


# # visualization

# In[6]:


sns.heatmap(df.isnull(),cbar=False)


# In[7]:


df=df[["PassengerId","Survived","Pclass","Age","SibSp","Parch","Fare","Sex","Embarked"]]


# In[9]:


df.head()


# In[10]:


sns.heatmap(df.isnull(),cbar=False)


# In[11]:


sns.catplot(y='Age',x='Pclass',kind='box',data=df)


# # replacing NaN values in 'Age'

# In[12]:


def fillage(a):
    for i in range(len(a["Fare"])):
        if a.iloc[i,3] not in range(100) and a.iloc[i,2]==1:
            a.iloc[i,3]=38
        elif a.iloc[i,3] not in range(100) and a.iloc[i,2]==2:
            a.iloc[i,3]=28
        elif a.iloc[i,3] not in range(100) and a.iloc[i,2]==3:
            a.iloc[i,3]=22    


# In[13]:


fillage(df)


# In[14]:


sns.heatmap(df.isnull(),cbar=False)


# In[15]:


df.head()


# # creating dummy variable

# In[16]:


df = pd.get_dummies(df, columns=["Sex","Embarked"])


# In[17]:


df.head()


# # preparing data

# In[18]:


X=df.drop('Survived',axis=1)


# In[19]:


Y=df['Survived']


# # ML model

# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33)


# In[22]:


from sklearn.linear_model import LogisticRegression


# In[23]:


model=LogisticRegression()


# In[24]:


model.fit(x_train,y_train)


# In[25]:


pred=model.predict(x_test)


# In[26]:


type(pred)


# In[27]:


from sklearn.metrics import confusion_matrix


# In[28]:


a=confusion_matrix(y_test,pred)


# In[29]:


print(a)


# In[30]:


from sklearn.metrics import classification_report


# In[31]:


print(classification_report(y_test,pred))


# In[32]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)


# In[33]:


pred=model.predict(x_test)


# In[34]:


print(classification_report(y_test,pred))


# In[35]:


from sklearn.ensemble import RandomForestClassifier


# In[36]:


model=RandomForestClassifier()
model.fit(x_train,y_train)


# In[37]:


pred=model.predict(x_test)


# In[38]:


print(classification_report(y_test,pred))


# In[ ]:




