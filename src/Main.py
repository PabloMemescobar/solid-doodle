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
from operator import itemgetter

# Constants
# Ensuring that everyone running the code can access the data
DATA_DIRECTORY = Path(os.path.dirname(os.getcwd()) + "/Data/")
TEXT_INDEX = 1

# Constants for figures
WIDTH = 10
HEIGHT = 4

# In[2]: Preparing data
# The functions and code in this section will help us import and prepare the data
# Within here, we run functions on our complete set of data that we later use to create visuals

with open(DATA_DIRECTORY / "Fake.csv", encoding = "latin1") as csvfile:
  fakeData = list(csv.reader(csvfile))

with open(DATA_DIRECTORY / "True.csv", encoding = "latin1") as csvfile:
  trueData = list(csv.reader(csvfile))

# This function will get the text column of the data, which is located in the 2nd column
# Other variables in the data - title, date & subject - are not needed in our program
def getText(data):
  text = []
  for i in range(1, len(data)):
      text.append([data[i][TEXT_INDEX]]) 
  return text

# This function will get the text column of the data and put it into a list with a label for whether the data is fake or true
# This list will be put into another list that will hold all [article, label] pairings
def getTextAddLabel(data, str):
  text = []
  varnames = data[0]
  for i in range(1, len(data)):
      text.append([data[i][TEXT_INDEX], str]) 
  return text

fakeData = getTextAddLabel(fakeData, "fake")
trueData = getTextAddLabel(trueData, "true")

# Combining the two lists of fake articles and real articles into one main data list
# This will be used for creating visuals located in In[10]
data = fakeData + trueData

# This function will create a list of all the labels of the articles
def getLabel(data):
  label_list = []
  for article in data:
    label_list.append(article[1])
  return label_list

label_list = getLabel(data)

# In[3]: Data Cleaning and Processing 
# This function will remove any empty columns from the data and put all words into lower case
# The data entering this function will be a list of lists (each article is list containing a single string)
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

# This function will loop through all words in the data and replace any "bad words" with a space
# Bad words is a custom list of words that we created that contain NLTK's stopwords, along with other words that we thought should be taken out 
# Within here, we run functions on our complete set of data that we later use to create visuals
# The data entering this function will be a list of lists (each article is list containing a single string)
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

# This function combines the previous two functions into one
# The data entering this function will be a list of lists (each article is list containing a single string)
def cleanData(data):
  data = removeColumnAndLowercase(data)
  data = replaceBadWords(data)

  return data

full_fakeData = cleanData(fakeData) 
full_trueData = cleanData(trueData)

full_data = fakeData + trueData

# In[4]: Make dictionaries

# This function takes in data and a list of labels (containing "true" or "fake") and returns a dictionary for both fake and true words
# The dictionary contains each word as a key and the number of times the word shows up in the respective articles as the value
# The data entering this function will be a list of lists (each article is list containing a single string)
# Before this step, the data is run through the cleaning functions
def make_dictionary(data, labels): 
  article_list = []
  for article in data:
    article_list.append(article[0].split()) 

  fake_dict = {}
  true_dict = {}
  i = 0
  while i < len(article_list):
    if labels[i] == "true":
        for word in article_list[i]:
            if word not in true_dict: 
                true_dict[word] = 0
            true_dict[word] += 1
    else:
        for word in article_list[i]:
            if word not in fake_dict:
                fake_dict[word] = 0
            fake_dict[word] += 1
    i += 1
  return fake_dict, true_dict

full_fake_dict, full_true_dict = make_dictionary(full_data, label_list)

# In[5]: Create all values needed for the Naive Bayes Classifier
# For the classifier, we want to take the dictionaries we made of the counts and turn those into probability dictionaries
# In this section, we made various functions that help to create probabilities

# This function adds together all values in dictionary 
# The purpose is so that we can get a count of the total number of cleaned words in both fake and true articles, respectively
# We use this in the next function to determine probabilities
def dict_count_words(dict):
    count = 0
    for key in dict:
        count += dict[key]
    return count

# This function creates dictionaries of probabilities of each word for both fake and true dictionaries
# Essentially, we want to know the probability of words showing up in fake articles and the probability of words showing up in true articles
def dict_create_probabilities(dict, count):
    prob_dict = {}
    for key in dict:
        prob_dict[key] = dict[key]/count
    return prob_dict

# The prupose of running the functions on the full data set is for the visualizations made in In[10], as mentioned before, but also to test that the functions work

# Create count of words for both fake and true articles for the full data set
full_fake_word_count = dict_count_words(full_fake_dict)
full_true_word_count = dict_count_words(full_true_dict)

# Create probability dictionaries for both fake and true articles for the full data set
full_fake_prob_dict = dict_create_probabilities(full_fake_dict, full_fake_word_count)
full_true_prob_dict = dict_create_probabilities(full_true_dict, full_true_word_count)

# Create probabilities of articles being fake or true using full data set
full_fake_prob = label_list.count("fake")/len(label_list)
full_true_prob = 1-full_fake_prob

# In[6]: Create Classifier
# In this section, we build the Bayes Classifier 
# articles refers to a list of lists where each list contains a cleaned article as 1 single string

# log(P(A|B)) = log(P(B|A)) + log(P(A))
# P(B|A) = P(word1 | true) * P(word2 | True) .... 

# This function looks at a list of articles
# For each article, the function splits the article so that each word is an individual string
# Then, the function looks for each of those words in the true probability dictionary to find the probability the word shows up in a true article
# If the word does not exist in the dictionary, the probability 1/count is assigned to the word, where count is the total number of words in true articles
# The function takes the log of this probability and adds it to a running count (conditional_true)
# Once all words have been accounted for, conditional_true which represent P(B|A) is multiplied by the probability of a true article P(A) and added to a list
# This function returns a list that contains P(true|B) for each article
def true_probability_list(articles):
  true_prob_list = [] 
  for article in articles:
    words = article[0].split()
    conditional_true = 0
    for word in words:
      if word in true_prob_dict:
        conditional_true += math.log(true_prob_dict[word]+(1/true_word_count)) # for each word in the article, we calculate P(word|true) and add the probabilities together (since they are logs)
      else:
        conditional_true += math.log(1/true_prob)
    true_prob_list.append(conditional_true+math.log(true_prob)) #log(P(B|A)) + log(P(A))
  return true_prob_list

# This function does the same as above, but for fake (replace each time we wrote true with fake)
def fake_probability_list(articles):
  fake_prob_list = [] 
  for article in articles:
    words = article[0].split()
    conditional_fake = 0 
    for word in words:
      if word in fake_prob_dict:
        conditional_fake += math.log(fake_prob_dict[word]+(1/fake_word_count)) # for each word in the article, we calculate P(word|fake) and add the probabilities together (since they are logs)
      else:
        conditional_fake += math.log(1/fake_prob)
    fake_prob_list.append(conditional_fake+math.log(fake_prob)) #P(B|A) * P(A)
  return fake_prob_list

# This function will allows us to test the accuracy of our program by also including the list of labels for each article (i.e. whether they're true or not)
# The function takes in the two lists from the two functions above and compares each index
# Whichever probability is higher, fake or real, the article will be assigned that label
# Once a label has been assigned to each article, we compare each index to the actual labels of the articles
# This function will return what the accuracy of the program was
# Additionally, we added code in that will return the percentage of false true articles and false fake articles we get
# False true means the program said it's true but it was actually fake, vice versa for false fake
def test_true_or_fake(true_probs, fake_probs, label_list): # takes in the lists created from the true_probability_list and fake_probability_list functions
    i = 1
    results = []
    while i < len(true_probs)+1: 
        if true_probs[i-1] > fake_probs[i-1]:
            results.append("true")
        else:
            results.append("fake")
        i += 1
    count = 0
    false_true = 0 
    false_fake = 0 
    i = 0
    while i < len(results): 
        if results[i] == label_list[i]:
            count += 1
        if results[i] == "true" and label_list[i] == "fake": # 
          false_true += 1
        if results[i] == "fake" and label_list[i] == "true":
          false_fake += 1
        i += 1
    print("The accuracy is", count/len(results), "False true is", false_true/len(label_list), "False fake is", false_fake/len(label_list))

# This function is like the one above, but here we assume that we do not know whether an article is true or fake
# Similar to above we compare each index of the probability lists from the first 2 functions and return whether the program decided the article is fake or true
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
# In this section, we prepare our data for cross validation

# Split articles into five random groups 
n = 5
random.seed(5)
random.shuffle(data)
five_split = np.array_split(data, n)

# This function will create article lists and label lists for each group
def makeLists(list): # Make 2 lists from list of lists containing 2 items (i.e. [article, label])
  article_list = []
  label_list = []
  for article in list:
    article_list.append([article[0]])
    label_list.append(article[1])
  return article_list, label_list

# Here, we are using the above function to make two lists for each of the five groups
# In each of the test lists, we have lists of articles
# We create the label list so that we can compare the results of the program with the actual labels
test1, test_label1 = makeLists(five_split[0])
test2, test_label2 = makeLists(five_split[1])
test3, test_label3 = makeLists(five_split[2])
test4, test_label4 = makeLists(five_split[3])
test5, test_label5 = makeLists(five_split[4])

# Here, each row is one set of training data that corresponds to the respective testing set
# For each testing set, the training set consists of all other groups of articles
# We also take the labels for each of these groups so that we can create dictionaries
train1, train_label1 = (test2 + test3 + test4 + test5), (test_label2 + test_label3 + test_label4 + test_label5)
train2, train_label2 = (test1 + test3 + test4 + test5), (test_label1 + test_label3 + test_label4 + test_label5)
train3, train_label3 = (test1 + test2 + test4 + test5), (test_label1 + test_label2 + test_label4 + test_label5)
train4, train_label4 = (test1 + test2 + test3 + test5), (test_label1 + test_label2 + test_label3 + test_label5)
train5, train_label5 = (test1 + test2 + test3 + test4), (test_label1 + test_label2 + test_label3 + test_label4)

# Create five sets of training/testing data
test_train = [[test1, test_label1, train1, train_label1], [test2, test_label2, train2, train_label2], [test3, test_label3, train3, train_label3], [test4, test_label4, train4, train_label4], [test5, test_label5, train5, train_label5]]

# In[8]: Cross Validation
# In this section, we run our entire program five times - once on each set of training/testing data

for set in test_train:
  # Create dictionaries using training data
  fake_dict, true_dict = make_dictionary(set[2], set[3])

  # Create count of words for both fake and true using training data
  fake_word_count = dict_count_words(fake_dict)
  true_word_count = dict_count_words(true_dict)

  # Create probability dictionaries for both fake and true using training data
  fake_prob_dict = dict_create_probabilities(fake_dict, fake_word_count)
  true_prob_dict = dict_create_probabilities(true_dict, true_word_count)

  # Create probabilities of article being fake or true using training data
  # P(A)
  fake_prob = set[3].count("fake")/len(label_list)
  true_prob = 1-fake_prob

  # Find probailities from classifier using the testing data
  fake_probs = fake_probability_list(set[0])
  true_probs = true_probability_list(set[0])

  # Determine accuracy of algorithm using the testing data and labels
  # This will give us the percentage of accuracy, as well as the percentage of false true and false fake
  test_true_or_fake(true_probs, fake_probs, set[1])

# In[9]: User input
# In this section, we created a function that takes in the file name from a user and outputs whether each article is real or fake according to the program
# Here, we use the entire data set as the training set
# It is assumed that the data is located in the same file as the python code, so we only ask for the file name
# We also assume the data is in the same format as the data used to build the program

user_input = input("Please enter your file name: ")

def classifier(user_input):
  with open(DATA_DIRECTORY / user_input, encoding = "utf8") as csvfile:
    user_data = list(csv.reader(csvfile))
  
  # Clean/process user data
  # Get only the text from the user input and clean it
  user_articles = getText(user_data)
  user_articles = cleanData(user_articles)
  
  # Create dictionaries using full data set
  fake_dict, true_dict = make_dictionary(full_data, label_list)

  # Create count of words for both fake and true using full data set
  fake_word_count = dict_count_words(fake_dict)
  true_word_count = dict_count_words(true_dict)

  # Create probability dictionaries for both fake and true using full data set
  fake_prob_dict = dict_create_probabilities(fake_dict, fake_word_count)
  true_prob_dict = dict_create_probabilities(true_dict, true_word_count)

  # Create probabilities of article being fake or true using full data set
  # P(A)
  fake_prob = label_list.count("fake")/len(label_list)
  true_prob = 1-fake_prob

  # Find probailities from classifier using user inputted data
  fake_probs = fake_probability_list(user_articles)
  true_probs = true_probability_list(user_articles)

  # Return the results of the program for each article - same order as in the data file
  true_or_fake(true_probs, fake_probs)

classifier(user_input)

# In[10]: Test holdout data
# Running program with holdout data as testing set and full data set as the training set

user_fake_data = "Fakeholdout.csv"
user_real_data = "Trueholdout.csv"

with open(DATA_DIRECTORY / user_fake_data, encoding = "latin1") as csvfile:
  fakeHoldout = list(csv.reader(csvfile))

with open(DATA_DIRECTORY / user_real_data, encoding = "latin1") as csvfile:
  trueHoldout = list(csv.reader(csvfile))

fakeHoldout = getText(fakeHoldout)
trueHoldout = getText(trueHoldout)

fakeHoldout = cleanData(fakeHoldout)
trueHoldout = cleanData(trueHoldout)

holdoutData = fakeHoldout + trueHoldout

holdout_label_list = ["fake"] * len(fakeHoldout) + ["true"] * len(trueHoldout)

fake_prob_dict = full_fake_prob_dict
true_prob_dict = full_true_prob_dict

fake_prob = full_fake_prob
true_prob = 1-fake_prob

true_probs = true_probability_list(holdoutData)
fake_probs = fake_probability_list(holdoutData)

test_true_or_fake(true_probs, fake_probs, holdout_label_list)

# In[11]: Visualizations
# This sections contains various visualizations that we created to explore our data set for the report and presentation

# TOP WORDS: Determine the top words in fake and true articles, respectively, and see how often those words show up in the other type of articles
# Creating dictionaries containing the top 10 proportion of words in true and false articles, respectively

# This function finds the top 10 words in a dictionary and returns a new dictionary only containing those values
def find_top_ten_words(dictionary):
    return dict(sorted(dictionary.items(), key = itemgetter(1), reverse = True)[:10])

# This function takes in two dictionaries, find the top 10 words in dict1 and then find the values of those word in dict2
def find_top_ten_word_in_other(dict1, dict2):
    top_ten_words_dict1 = dict(sorted(dict1.items(), key = itemgetter(1), reverse = True)[:10])
    top_ten_words_in_dict1_in_dict2 = {}
    for key in top_ten_words_dict1:
        if key in dict2:
            top_ten_words_in_dict1_in_dict2[key] = dict2[key]
    return top_ten_words_in_dict1_in_dict2

# Top 10 words in fake probability dictionary
top_ten_fake_prop_dict = find_top_ten_words(full_fake_prob_dict)
# Top 10 words in true probability dictionary
top_ten_true_prop_dict = find_top_ten_words(full_true_prob_dict)

# Probabilities of the top 10 words from fake articles in true articles
top_ten_fake_prop_in_true_dict = find_top_ten_word_in_other(full_true_prob_dict, full_fake_prob_dict) 
# Probabilities of the top 10 words from true articles in fake articles
top_ten_true_prop_in_fake_dict = find_top_ten_word_in_other(full_fake_prob_dict, full_true_prob_dict) 

# Plot of top 10 words in true articles and their proportions in both fake and true articles
top_ten_true_words = top_ten_true_prop_dict.keys()
X_axis = np.arange(len(top_ten_true_words))

plt.figure(figsize=(WIDTH, HEIGHT))
plt.bar(X_axis-0.2, top_ten_true_prop_dict.values(), width = 0.4, label = "True")
plt.bar(X_axis+0.2, top_ten_fake_prop_in_true_dict.values(), width = 0.4, label = "Fake")

plt.xticks(X_axis, top_ten_true_words)
plt.xlabel("Words")
plt.ylabel("Proportion of Words in Articles")
plt.title("Proportion of Top 10 Words of True News in Articles")
plt.legend()
plt.show()

# Plot of top 10 words in fake articles and their proportions in both fake and true articles
top_ten_fake_words = top_ten_fake_prop_dict.keys()
X_axis = np.arange(len(top_ten_fake_words))

plt.figure(figsize=(WIDTH, HEIGHT))
plt.bar(X_axis+0.2, top_ten_true_prop_in_fake_dict.values(), width = 0.4, label = "True")
plt.bar(X_axis-0.2, top_ten_fake_prop_dict.values(), width = 0.4, label = "Fake")

plt.xticks(X_axis, top_ten_fake_words)
plt.xlabel("Words")
plt.ylabel("Proportion of Words in Articles")
plt.title("Proportion of Top 10 Words of Fake News in Articles")
plt.legend()
plt.show()


# ACCURACY: Just creating a visual for our cross validation for the presentation
sets = ["First", "Second", "Third", "Fourth", "Fifth"]
values = [0.7111, 0.6990, 0.7093, 0.7167, 0.7151]
X_axis = np.arange(len(sets))

plt.figure(figsize=(WIDTH, HEIGHT))

width = 0.75
fig, ax = plt.subplots()

ax.set_ylabel('Accuracy')
ax.set_xlabel('Training/Testing Set')
ax.set_title('Cross Validation of Naive Bayes Classifier')
ax.set_xticks(X_axis)
ax.set_xticklabels(sets)
ax.set_ylim([0, 0.80])

pps = ax.bar(X_axis, values, width, label='values')
for p in pps:
   height = p.get_height()
   ax.annotate('{}'.format(height),
      xy=(p.get_x() + p.get_width() / 2, height),
      xytext=(0, 3), 
      textcoords="offset points",
      ha='center', va='bottom')
plt.show()

# %%
