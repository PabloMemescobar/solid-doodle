#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#shuffle
from sklearn.utils import shuffle
#removing number and special character
import re
#remove stopwords 
import nltk
from nltk.corpus import stopwords


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
# Merge "fake" and "true" datasets
data = pd.concat([df_f, df_t]).reset_index(drop = True)


# In[5]:


#shuffle
data = data.sample(frac=1)
data.head()


# In[6]:


#df.apply(lambda x: func(x['col1'],x['col2']),axis=1)
#convert "text" column to all lowercase letters
data['text'] = data['text'].apply(lambda x: x.lower())
data.head()


# In[7]:


#removing stopwords
nltk.download('stopwords')
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#removing number and special character
def cleaning(text):
    text = re.sub('[^a-z]' , ' ', text)
    return text
data['text'] = data['text'].apply(cleaning)
data.head()


# In[ ]:




