# In[1]: Import packages

#Modules
import os
import csv
from pathlib import Path
import math
from sklearn.model_selection import train_test_split
from badWords import badWordsTable
import re; pattern = re.compile('[^a-zA-Z]+')
import time
import multiprocessing
from multiprocessing import Pool
from timeit import default_timer as timer
import sys
import random
import numpy as np
import pandas as pd

# Constants
# Ensuring that everyone running the code can access the data
DATA_DIRECTORY = Path(os.path.dirname(os.getcwd()) + "/Data/")

# In[2]: Preparing data
with open(DATA_DIRECTORY / "Fake.csv", encoding = "utf8") as csvfile:
    fakeData = list(csv.reader(csvfile))

with open(DATA_DIRECTORY / "True.csv", encoding = "utf8") as csvfile:
  trueData = list(csv.reader(csvfile))

def getText(data):
  text = []
  for i in range(1, len(data)):
      text.append([data[i][1]]) # The article text is located at [1] (second column in the data)
  return text

def getTextAddLabel(data, str):
  text = []
  for i in range(1, len(data)):
      text.append([data[i][1], str]) # The article text is located at [1] (second column in the data)
  return text

fakeData = getTextAddLabel(fakeData, "fake")
trueData = getTextAddLabel(trueData, "true")

data = fakeData + trueData

def getLabel(data):
  label_list = []
  for article in data:
    label_list.append(article[1])
  return label_list

label_list = getLabel(data)

# In[3]: Data Cleaning and Processing 
def removeColumnAndLowercase(data):
  counter = 0
  for articles in data:
    for i, words in enumerate(articles):
      articles[i] = articles[i].lower()
      counter+=1
      if (len(words) == 0):
        del articles[counter-1:]
        counter = 0
        break

  return data

def replaceBadWords(data):
  singleWordTable = ""
  for articles in data:
    for i, words in enumerate(articles):
      words = pattern.sub(' ', words)
      for singleWord in words:
        if singleWord == " ":
          if len(singleWordTable) <= 3:
            words = words.replace(" " + singleWordTable + " ", " ")
            singleWordTable = ""
            continue
          if singleWordTable in badWordsTable:
            words = words.replace(" " + singleWordTable + " ", " ")
          singleWordTable = ""
          continue
        singleWordTable += singleWord
      articles[i] = words

  return data

def cleanData(data):
  data = removeColumnAndLowercase(data)
  data = replaceBadWords(data)

  return data

fakeData = cleanData(fakeData)
trueData = cleanData(trueData)

data = fakeData + trueData

# In[4]: Make dictionaries

def makeDictionary(data, labels): #Data is a list of lists containing a single string with the cleaned text
  article_list = []
  for article in data:
    article_list.append(article[0].split()) #Turn each single string article into a list of strings for each word

  fake_dict = {}
  true_dict = {}
  i = 0
  while i < len(article_list):
    if labels[i] == "true":
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
  return fake_dict, true_dict

fake_dict, true_dict = makeDictionary(data, label_list)

# In[5]: Create all values needed for the Naive Bayes Classifier

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

#create probability dictionaries for both fake and true
fake_prob_dict = dict_create_probabilities(fake_dict, fake_word_count)
true_prob_dict = dict_create_probabilities(true_dict, true_word_count)

#create probabilities of article being fake or true using testing data
fake_prob = label_list.count("fake")/len(label_list)
true_prob = 1-fake_prob

# In[6]: Create Classifier
# articles is a list of lists where each list contains a cleaned article as 1 single string

# P(A|B) = P(B|A)*P(A)
# P(B|A) = P(word1 | true) * P(word2 | True) .... 

def true_probability_list(articles):
  true_prob_list = [] #P(true|B) for each article
  for article in articles:
    words = article[0].split()
    conditional_true = 0 #P(B|A)
    for word in words:
      if word in true_prob_dict:
        conditional_true += math.log(true_prob_dict[word]+(1/true_word_count)) # for each word in the article, we calculate P(word|true) and add the probabilities together (since they are logs)
      else:
        conditional_true += math.log(1/true_prob)
    true_prob_list.append(conditional_true+math.log(true_prob)) #log(P(B|A)) + log(P(A))
  return true_prob_list

def fake_probability_list(articles):
  fake_prob_list = [] #P(fake|B) for each article
  for article in articles:
    words = article[0].split()
    conditional_fake = 0 #P(B|A)
    for word in words:
      if word in fake_prob_dict:
        conditional_fake += math.log(fake_prob_dict[word]+(1/fake_word_count)) # for each word in the article, we calculate P(word|fake) and add the probabilities together (since they are logs)
      else:
        conditional_fake += math.log(1/fake_prob)
    fake_prob_list.append(conditional_fake+math.log(fake_prob)) #P(B|A) * P(A)
  return fake_prob_list

# this function will allow us to test the accuracy of our program by also including the list of labels for each article (i.e. whether they're true or not)
def test_true_or_fake(true_probs, fake_probs, label_list): # takes in the lists created from the true_probability_list and fake_probability_list functions
    i = 1
    results = []
    while i < len(true_probs)+1: # compares each index in both lists and determines which value is bigger (i.e. if P(true|article) > P(fake|article) then return "Article i is True")
        if true_probs[i-1] > fake_probs[i-1]:
            results.append("true")
        else:
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

# In[7]: Creating testing and training data

# Split articles into five groups
n = 5
random.shuffle(data)
five_split = np.array_split(data, n)

# Create article list and label list for each group
def makeLists(list): # Make 2 lists from list of list containing 2 items
  article_list = []
  label_list = []
  for article in list:
    article_list.append([article[0]])
    label_list.append(article[1])
  return article_list, label_list
    
test1, test_label1 = makeLists(five_split[0])
test2, test_label2 = makeLists(five_split[1])
test3, test_label3 = makeLists(five_split[2])
test4, test_label4 = makeLists(five_split[3])
test5, test_label5 = makeLists(five_split[4])

train1, train_label1 = (test2 + test3 + test4 + test5), (test_label2 + test_label3 + test_label4 + test_label5)
train2, train_label2 = (test1 + test3 + test4 + test5), (test_label1 + test_label3 + test_label4 + test_label5)
train3, train_label3 = (test1 + test2 + test4 + test5), (test_label1 + test_label2 + test_label4 + test_label5)
train4, train_label4 = (test1 + test2 + test3 + test5), (test_label1 + test_label2 + test_label3 + test_label5)
train5, train_label5 = (test1 + test2 + test3 + test4), (test_label1 + test_label2 + test_label3 + test_label4)

# Create five sets of train/test data
test_train = [[test1, test_label1, train1, train_label1], [test2, test_label2, train2, train_label2], [test3, test_label3, train3, train_label3], [test4, test_label4, train4, train_label4], [test5, test_label5, train5, train_label5]]

# In[8]: Cross Validation

for set in test_train:
  #create dictionaries
  fake_dict, true_dict = makeDictionary(set[2], set[3])

  #create count of words for both fake and true
  fake_word_count = dict_count_words(fake_dict)
  true_word_count = dict_count_words(true_dict)

  #create probability dictionaries for both fake and true
  fake_prob_dict = dict_create_probabilities(fake_dict, fake_word_count)
  true_prob_dict = dict_create_probabilities(true_dict, true_word_count)

  #create probabilities of article being fake or true using testing data
  fake_prob = set[3].count("fake")/len(label_list)
  true_prob = 1-fake_prob

  #find probailities from classifier
  true_probs = true_probability_list(set[0])
  fake_probs = fake_probability_list(set[0])

  #determine accuracy of algorithm
  test_true_or_fake(true_probs, fake_probs, set[1])

# In[9]: User input

user_input = input("Please enter file path: ")

# THIS NEEDS TO BE WORKED ON STILL - WILL BE UPDATED NOV 30
# def classifier(user_input, data):
  # articles = getText(data)
  # articles = cleanData(articles)
  
  # full_data = getText(data)

  # fake_dict, true_dict = makeDictionary(full_data, label_list)

  #create count of words for both fake and true
  # fake_word_count = dict_count_words(fake_dict)
  # true_word_count = dict_count_words(true_dict)

  #create probability dictionaries for both fake and true
  # fake_prob_dict = dict_create_probabilities(fake_dict, fake_word_count)
  # true_prob_dict = dict_create_probabilities(true_dict, true_word_count)

  #create probabilities of article being fake or true using testing data
  # fake_prob = label_list.count("fake")/len(label_list)
  # true_prob = 1-fake_prob

  #find probailities from classifier
  # true_probs = true_probability_list(articles)
  # fake_probs = fake_probability_list(articles)

  #determine accuracy of algorithm
  # true_or_fake(true_probs, fake_probs)

