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


# In[13]
#create all values needed for the Naive Bayes Classifier

#function that counts adds together all values in dictionary
def dict_count_words(dict):
    count = 0
    for key in dict:
        count += dict[key]
    return count

#create dictionaries of probabilities of each word for both fake and true dictionaries
def dict_create_probabilities(dict, count):
    prob_dict = {}
    for key in dict:
        prob_dict[key] = dict[key]/count
    return prob_dict

#create count of words for both fake and true
fake_word_count = dict_count_words(fake_dict)
true_word_count = dict_count_words(true_dict)

avg_words = (fake_word_count + true_word_count) / 2

#create probability dictionaries for both fake and true
fake_prob_dict = dict_create_probabilities(fake_dict, fake_word_count)
true_prob_dict = dict_create_probabilities(true_dict, true_word_count)

#create probabilities of article being fake or true using testing data
fake_prob = label_list.count("fake")/len(label_list)
true_prob = 1-fake_prob

#test
print(fake_dict["trump"]/fake_word_count)
print(fake_prob_dict["trump"])
print(true_dict["trump"]/true_word_count)
print(true_prob_dict["trump"])

print(fake_dict["president"]/fake_word_count)
print(fake_prob_dict["president"])
print(true_dict["president"]/true_word_count)
print(true_prob_dict["president"])


#In[14]
#create naive bayes classifier using dictionaries created above

# Class NaiveBayesClassifier(file): #assuming the file would be the cleaned file that we run through our cleaning function
# article_list = file.tolist()

import math

def true_probability_list(articles):
    true_prob_list = []
    for article in articles:
        conditional_true = 0 #P(B|A)
        for word in article:
            if word in true_prob_dict:
                conditional_true += math.log(true_prob_dict[word]) # for each word in the article, we calculate P(word|true) and add the probabilities together (since they are logs)
            #else:
                #conditional_true += math.log(10000/true_prob)
        true_prob_list.append(conditional_true*true_prob) #P(B|A) * P(A)
    return true_prob_list

def fake_probability_list(articles):
    fake_prob_list = []
    for article in articles:
        conditional_fake = 0 #P(B|A)
        for word in article:
            if word in fake_prob_dict:
                conditional_fake += math.log(fake_prob_dict[word]) # for each word in the article, we calculate P(word|fake) and add the probabilities together (since they are logs)
            #else:
                #conditional_fake += math.log(10000/fake_prob)
        fake_prob_list.append(conditional_fake*fake_prob) #P(B|A) * P(A)
    return fake_prob_list

# this function will allow us to test the accuracy of our program by also including the list of labels for each article (i.e. whether they're true or not)
def test_true_or_fake(true_probs, fake_probs, label_list): # takes in the lists created from the true_probability_list and fake_probability_list functions
    i = 1
    results = []
    while i < len(true_probs)+1: # compares each index in both lists and determines which value is bigger (i.e. if P(true|article) > P(fake|article) then return "Article i is True")
        if true_probs[i-1] > fake_probs[i-1]:
            print("Article " + str(i) + " is True.")
            results.append("true")
        else:
            print("Article " + str(i) + " is Fake.")
            results.append("fake")
        i += 1
    
    count = 0
    i = 0
    while i < len(results): # Here we are comparing the list inputed with the list created in the program to determine the accuracy
        if results[i] == label_list[i]:
            count += 1
        i += 1
    print("The accuracy is", count/len(results))

def true_or_fake(true_probs, fake_probs):
    while i in range(len(true_probs)+1):
        results = []
        if true_probs[i-1] > fake_probs[i-1]:
            print("Article " + str(i) + " is True.")
            results.append("true")
        else:
            print("Article " + str(i) + " is Fake.")
            results.append("true")


# %%

#test
test_articles = X_test.tolist()
labels = y_test.tolist()

test_true_or_fake(true_probability_list(test_articles), fake_probability_list(test_articles), labels)
# %%

# 0.0.5739529706414359 using logs, but before adding in else statement
# 0.49019062195631 using logs and adding in else statement with 1/prob

