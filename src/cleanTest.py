# In[1]: Import packages

#Modules
import os
import csv
from pathlib import Path
import math
from sklearn.model_selection import train_test_split
from badWords import badWordsTable
import re; pattern = re.compile('[^a-zA-Z]+')
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

full_fakeData = cleanData(fakeData)
full_trueData = cleanData(trueData)

full_data = fakeData + trueData

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

full_fake_dict, full_true_dict = makeDictionary(full_data, label_list)

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
full_fake_word_count = dict_count_words(full_fake_dict)
full_true_word_count = dict_count_words(full_true_dict)

#create probability dictionaries for both fake and true
full_fake_prob_dict = dict_create_probabilities(full_fake_dict, fake_word_count)
full_true_prob_dict = dict_create_probabilities(full_true_dict, true_word_count)

#create probabilities of article being fake or true using testing data
full_fake_prob = label_list.count("fake")/len(label_list)
full_true_prob = 1-full_fake_prob


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
    accuracy = []
    i = 0
    while i < len(results): # Here we are comparing the list inputed with the list created in the program to determine the accuracy
        if results[i] == label_list[i]:
            count += 1
        i += 1
    print("The accuracy is", count/len(results))

def true_or_fake(true_probs, fake_probs):
    for i in range(1, len(true_probs)+1):
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
random.seed(5)
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

  #create probabilities of article being fake or true using training data
  #P(A)
  fake_prob = set[3].count("fake")/len(label_list)
  true_prob = 1-fake_prob

  #find probailities from classifier
  true_probs = true_probability_list(set[0])
  fake_probs = fake_probability_list(set[0])

  #determine accuracy of algorithm
  test_true_or_fake(true_probs, fake_probs, set[1])

# In[9]: User input

user_input = input("Please enter file name: ")

def classifier(user_input):
  with open(DATA_DIRECTORY / user_input, encoding = "utf8") as csvfile:
    user_data = list(csv.reader(csvfile))
  
  user_articles = getText(user_data)
  user_articles = cleanData(user_articles)
  
  #create dictionaries
  fake_dict, true_dict = makeDictionary(full_data, label_list)

  #create count of words for both fake and true
  fake_word_count = dict_count_words(fake_dict)
  true_word_count = dict_count_words(true_dict)

  #create probability dictionaries for both fake and true
  fake_prob_dict = dict_create_probabilities(fake_dict, fake_word_count)
  true_prob_dict = dict_create_probabilities(true_dict, true_word_count)

  #create probabilities of article being fake or true using training data
  #P(A)
  fake_prob = label_list.count("fake")/len(label_list)
  true_prob = 1-fake_prob

  #find probailities from classifier
  true_probs = true_probability_list(user_articles)
  fake_probs = fake_probability_list(user_articles)

  #determine accuracy of algorithm
  true_or_fake(true_probs, fake_probs)

classifier(user_input)

# In[10]: Visualizations

# TOP WORDS

#creating dictionary containing the top 10 proportion of words in true and false articles, respectively
from operator import itemgetter

#this function find the 10 highest values in a dictionary and returns a new dictionary only containing those values
def find_top_10(dictionary):
    return dict(sorted(dictionary.items(), key = itemgetter(1), reverse = True)[:10])

#this function takes in two dictionaries, find the top 10 words in dict1 and then find the proportion of those word in dict2
def find_top_word_in_other(dict1, dict2):
    top10_word_prop_in_other_dict = {}
    for key in find_top_10(dict1):
        if key in dict2:
            top10_word_prop_in_other_dict[key] = dict2[key]
    return top10_word_prop_in_other_dict

top10_fake_prop_dict = find_top_10(full_fake_prob_dict)
top10_true_prop_dict = find_top_10(full_true_prob_dict)

top10_fake_prop_in_true_dict = find_top_word_in_other(full_true_prob_dict, full_fake_prob_dict)
top10_true_prop_in_fake_dict = find_top_word_in_other(full_fake_prob_dict, full_true_prob_dict)

X = top10_true_prop_dict.keys()
X_axis = np.arange(len(X))

plt.figure(figsize=(10, 4))
plt.bar(X_axis-0.2, top10_true_prop_dict.values(), width = 0.4, label = "True")
plt.bar(X_axis+0.2, top10_fake_prop_in_true_dict.values(), width = 0.4, label = "Fake")

plt.xticks(X_axis, X)
plt.xlabel("Words")
plt.ylabel("Proportion of Words in Articles")
plt.title("Proportion of Top 10 Words of True News in Articles")
plt.legend()
plt.show()

#plot of top 10 words in fake and their proportions in both fake and true articles
X = top10_fake_prop_dict.keys()
X_axis = np.arange(len(X))

plt.figure(figsize=(10, 4))
plt.bar(X_axis+0.2, top10_true_prop_in_fake_dict.values(), width = 0.4, label = "True")
plt.bar(X_axis-0.2, top10_fake_prop_dict.values(), width = 0.4, label = "Fake")

plt.xticks(X_axis, X)
plt.xlabel("Words")
plt.ylabel("Proportion of Words in Articles")
plt.title("Proportion of Top 10 Words of Fake News in Articles")
plt.legend()
plt.show()


# ACCURACY
X = ["First", "Second", "Third", "Fourth", "Fifth"]
values = [0.7111, 0.6990, 0.7093, 0.7167, 0.7151]

plt.figure(figsize=(10, 4))

x = np.arange(len(X)) # the label locations
width = 0.75 # the width of the bars

fig, ax = plt.subplots()

ax.set_ylabel('Accuracy')
ax.set_xlabel('Training/Testing Set')
ax.set_title('Cross Validation of Naive Bayes Classifier')
ax.set_xticks(x)
ax.set_xticklabels(X)
ax.set_ylim([0, 0.80])

pps = ax.bar(x, values, width, label='values')
for p in pps:
   height = p.get_height()
   ax.annotate('{}'.format(height),
      xy=(p.get_x() + p.get_width() / 2, height),
      xytext=(0, 3), # 3 points vertical offset
      textcoords="offset points",
      ha='center', va='bottom')
plt.show()

# %%
