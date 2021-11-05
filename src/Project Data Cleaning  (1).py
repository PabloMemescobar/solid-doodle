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
#vectorize the words to numerical data 
from sklearn.feature_extraction.text import TfidfVectorizer
#split the data into train and test
from sklearn.model_selection import train_test_split


# In[2]:


#import dataset "True"
df_t=pd.read_csv('/Users/erinliang/Desktop/411 python/True.csv')
df_t.head()


# In[3]:


#import dataset "Fake"
df_f=pd.read_csv('/Users/erinliang/Desktop/411 python/Fake.csv')
df_f.drop(df_f.columns[4:129],axis=1,inplace=True)
df_f.head()


# In[4]:


#add label column
df_f['label'] = 'fake'
df_t['label'] = 'true'
#merge "fake" and "true" datasets
#pandas.concat takes a list or dict and concatenates them into one
#'.reset_index(drop = True)': delete the index instead of inserting it back into the columns of the DataFrame
data = pd.concat([df_f, df_t]).reset_index(drop = True)


# In[5]:


#shuffle
#frac: the fraction of rows to return in the random sample, in this case 100%
data = data.sample(frac=1)
data.head()


# In[6]:


#apply() function calls the lambda function and applies it to a Pandas series
#convert "text" column to all lowercase letters
data['text'] = data['text'].apply(lambda x: x.lower())
data.head()


# In[7]:


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


# In[8]:


#vectorize the words to numerical data 
x = TfidfVectorizer()
x.fit(data['text'])
x = x.transform(data['text'])
print(x.toarray())


# In[ ]:




