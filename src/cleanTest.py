# In[1]:

#Modules
import os
import csv
from pathlib import Path
import math
import re
from sklearn.model_selection import train_test_split
from badWords import badWordsTable
import re; pattern = re.compile('[^a-zA-Z_]+')
import time
from multiprocessing import Pool

# Constants
DATA_DIRECTORY = Path(os.path.dirname(os.getcwd()) + "/Data/")

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

if __name__=="__main__":
  with open(DATA_DIRECTORY / "Fake.csv", encoding = "utf8") as csvfile:
    fakeData = list(csv.reader(csvfile))

  with open(DATA_DIRECTORY / "True.csv", encoding = "utf8") as csvfile:
    trueData = list(csv.reader(csvfile))

  data = fakeData + trueData
  masterDataTable = []

  chunks = [data[x:x+10] for x in range(0, len(data), 10)]

  with Pool(processes=10) as pool:
    threadData = [pool.apply_async(cleanData, (chunks[i],)) for i in range(len(chunks))]
    masterDataTable+=[sections.get(timeout=100) for sections in threadData]

    result = pool.apply_async(time.sleep, (10,))

  #show articles
  # for articles in masterDataTable:
  #   print(articles)
  #   print("\n \n \n")

  #split the data into train and test samples - sklearn
  #20% of the dataset will be use to test; 'random_state' could be any number

  # Xdata, Ydata = np.arange(10).reshape((5, 2)), range(5)

  X_train, X_test, y_train, y_test = train_test_split(masterDataTable, masterDataTable, test_size=0.2, random_state=88)

  # #creating a two dictionaries; one that contains all words in real articles and one that contains all words in fake articles
  true_dict = {}
  fake_dict = {}

  #turning the dataframes into lists to index
  label_list = y_train
  articles = X_train

  #to be able to loop through each word in the article, we need to take the list of strings and turn it into a list that holds a list of words for each article
  article_list = []

  for article in masterDataTable:
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
              # print("Article " + str(i) + " is True.")
              results.append("true")
          else:
              # print("Article " + str(i) + " is Fake.")
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
              # print("Article " + str(i) + " is True.")
              results.append("true")
          else:
              # print("Article " + str(i) + " is Fake.")
              results.append("true")

  test_articles = X_test
  labels = y_test
  # print(type(test_articles))
  # print(type(true_probability_list(test_articles)))
  # cleaning_data(test_articles)
  test_true_or_fake(true_probability_list(test_articles), fake_probability_list(test_articles), labels)

  # 0.0.5739529706414359 using logs, but before adding in else statement
  # 0.49019062195631 using logs and adding in else statement with 1/prob

