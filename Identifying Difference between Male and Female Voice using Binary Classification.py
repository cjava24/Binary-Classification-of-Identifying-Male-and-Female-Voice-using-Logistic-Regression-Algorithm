#!/usr/bin/env python
# coding: utf-8

# In[22]:

# import the necessary libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# In[6]:

# load the dataset and look at the data
data = pd.read_csv("voice.csv", header = 0)
data
print(data.head(6))

data.info()


# In[8]:


# look at columns
data.columns


# In[12]:

# setup the different labels for logistic regression
data['label'] = data['label'].map({'male': 1, 'female': 0})


# In[13]:

# describe the data
data.describe()


# In[14]:

# plot the count plot
sns.countplot(data['label'], label = "Count")
plt.show()


# In[15]:

# plot the heatmap
corr = data.corr()
plt.figure(figsize = (4,4))
sns.heatmap(corr, cbar = True, square = True, cmap = "coolwarm")
plt.show()


# In[16]:

# declare the prediction variables
prediction_var = ['median','Q25', 'IQR', 'skew', 'sfm']


# In[18]:

# split the training and testing data sets
train,test = train_test_split(data,test_size = 0.2)
print(train.shape)
print(test.shape)


# In[19]:

# Setup training and testing variables
train_X = train[prediction_var]
train_y = train.label
test_X = test[prediction_var]
test_y = test.label


# In[20]:

# Perform Logistic Regression Algorithm
logistic = LogisticRegression()
logistic.fit(train_X, train_y)
control = logistic.predict(test_X)
print(metrics.accuracy_score(control, test_y))


# In[23]:

# Perform analysis using Decision Tree Algorithm
clf = DecisionTreeClassifier(random_state = 0)
cross_val_score(clf, train_X, train_y, cv = 10)
clf.fit(train_X, train_y,sample_weight = None, check_input = True, X_idx_sorted = None)
clf.get_params(deep = True)
clf.predict(test_X, check_input = True)
clf.predict_log_proba(test_X)
clf.predict(test_X, check_input = True)
print(clf.score(test_X, test_y, sample_weight = None))

