#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#shuffle
from sklearn.utils import shuffle

#remove numbers and special characters
import re

#remove stopwords 
import nltk
from nltk.corpus import stopwords

#split the data into train and test samples
from sklearn.model_selection import train_test_split


# In[2]:


#import datasets 
fakeData = DATA_DIRECTORY / "Fake.csv"
trueData = DATA_DIRECTORY / "True.csv"


# In[3]:


#add label column
fakeData['label'] = 'fake'
trueData['label'] = 'true'
#merge "fake" and "true" datasets
#pandas.concat takes a list or dict and concatenates them into one
#'.reset_index(drop = True)': delete the index instead of inserting it back into the columns of the DataFrame
data = pd.concat([fakeData, trueData]).reset_index(drop = True)


# In[4]:


#shuffle dataset for training and testing purposes 
#frac: the fraction of rows to return in the random sample, in this case 100%
data = data.sample(frac=1)
data.head()


# In[5]:


#convert "text" column to all lowercase letters
#apply() function calls the lambda function and applies it to a Pandas series
data['text'] = data['text'].apply(lambda x: x.lower())
data.head()


# In[6]:


#for removing stopwords
nltk.download('stopwords')
stop = stopwords.words('english')

#loop every element in x.split() and create a new list that contains only the elements that meet the if condition
#and join all items in the list into ONE string, using space as separator
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#remove numbers and special characters
def cleaning(text):
    text = re.sub('[^a-z]' , ' ', text)
    return text
data['text'] = data['text'].apply(cleaning)
data.head()


# In[11]:


#split the data into train and test samples - sklearn
#20% of the dataset will be use to test; 'random_state' could be any number 
X_train,X_test,y_train,y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=88)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

