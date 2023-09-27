#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[3]:


df=pd.read_csv("C:/Users/Lenovo/Classification+preprocessed+data+Python.csv",header=0)


# In[4]:


df.head()


# # Logistic Regression Model

# In[6]:


x=df[['price']]#independent variable
y=df['Sold']


# In[7]:


x.head()


# In[8]:


y.head()


# In[10]:


from sklearn.linear_model import LogisticRegression


# In[11]:


clf_lrs =LogisticRegression()


# In[12]:


clf_lrs.fit(x,y)


# In[14]:


clf_lrs.coef_


# In[16]:


clf_lrs.intercept_


# In[18]:


import statsmodels.api as sn


# In[19]:


x_cons=sn.add_constant(x)


# In[20]:


x_cons.head()


# In[21]:


import statsmodels.discrete.discrete_model as sm


# In[22]:


logit = sm.Logit(y,x_cons).fit()


# In[23]:


logit.summary()


# # Logistic Regression with multiple predictor

# In[24]:


x=df.loc[:,df.columns!='Sold']


# In[25]:


y=df['Sold']


# In[32]:


clf_lr =LogisticRegression()
clf_lr.fit(x,y)


# In[33]:


clf_lr.coef_


# In[34]:


clf_lr.intercept_


# In[29]:


x_cons=sn.add_constant(x)
logit = sm.Logit(y,x_cons).fit()


# In[30]:


logit.summary()


# # Predicting and confusion matrix

# In[37]:


clf_lr.predict_proba(x)


# In[38]:


y_pred=clf_lr.predict(x)
y_pred


# In[40]:


y_pred_03=clf_lr.predict_proba(x)[:,1]>=0.3
y_pred_03


# In[41]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y,y_pred)


# In[42]:



confusion_matrix(y,y_pred_03)


# # Performance metrics

# In[45]:


from sklearn.metrics import precision_score, recall_score


# In[46]:


precision_score(y,y_pred)


# In[47]:


recall_score(y,y_pred)


# In[48]:


from sklearn.metrics import roc_auc_score


# In[50]:


roc_auc_score(y,y_pred)

