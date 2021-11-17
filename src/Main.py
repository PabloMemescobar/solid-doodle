#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from pathlib import Path

#shuffle
from sklearn.utils import shuffle

#remove numbers and special characters
import re

#remove stopwords
import stop_words
from stop_words import get_stop_words

#split the data into train and test samples
from sklearn.model_selection import train_test_split


# In[2]:


# Constants
DATA_DIRECTORY = Path(os.path.dirname(os.getcwd()) + "/Data/")

#import datasets

fakeData = pd.read_csv(DATA_DIRECTORY / "Fake.csv")
trueData = pd.read_csv(DATA_DIRECTORY / "True.csv")

# drop the extra columns in fakeData
cols = list(range(4, 129))
fakeData.drop(fakeData.columns[cols], axis=1, inplace=True)


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


# In[5]:


#convert "text" column to all lowercase letters
#apply() function calls the lambda function and applies it to a Pandas series
data['text'] = data['text'].apply(lambda x: x.lower())


# In[6]:


#for removing stopwords
stop = get_stop_words('en')

#loop every element in x.split() and create a new list that contains only the elements that meet the if condition
#and join all items in the list into ONE string, using space as separator
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#remove numbers and special characters
def cleaning(text):
    text = re.sub('[^a-z]' , ' ', text)
    return text
data['text'] = data['text'].apply(cleaning)


# In[11]:


#split the data into train and test samples - sklearn
#20% of the dataset will be use to test; 'random_state' could be any number 
X_train,X_test,y_train,y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=88)


# In[12]


#creating a two dictionaries; one that contains all words in real articles and one that contains all words in fake articles
true_dict = {}
fake_dict = {}

#turning the dataframes into lists to index
label_list = y_train.tolist()
articles = X_train.tolist()

#to be able to loop through each word in the article, we need to take the list of strings and turn it into a list that holds a list of words for each article
article_list = []

for article in articles:
    article_list.append(article.split())


#loops through each article, check whether its true or false to put the words in correct dictionary, and then loops through the list of words
i = 0
while i < len(articles):
    if label_list[i] == "true":
        for word in article_list[i]:
            if word not in true_dict: #create an entry in the dict if it doesn't exist already
                true_dict[word] = 0
            true_dict[word] += 1
    else:
        for word in article_list[i]:
            if word not in fake_dict:
                fake_dict[word] = 0
            fake_dict[word] += 1
    i += 1

#update: update dictionaries to only contain words with length creater than 3

new_true_dict = {} #temp dict
new_fake_dict = {} #temp dict

for key in true_dict:
    if len(key) > 3:
        new_true_dict[key] = true_dict[key]

for key in fake_dict:
    if len(key) > 3:
        new_fake_dict[key] = fake_dict[key]

true_dict = new_true_dict
fake_dict = new_fake_dict

print(len(true_dict))
print(len(fake_dict))


# In[13]


#function to clean data
def cleanData(text):
    text = str(text).lower()
    text = re.sub('[^a-z]' , ' ', text)
    stop = get_stop_words('en')
    text = ' '.join([word for word in text.split() if word not in (stop)])
    return text
