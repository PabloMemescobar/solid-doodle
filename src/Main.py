import os
import csv

# Path to the data
DATA_DIRECTORY = os.path.dirname(os.getcwd()) + "\\Data"

# Fake data and real data files 
fakeData = DATA_DIRECTORY + "\\Fake.csv"
trueData = DATA_DIRECTORY + "\\True.csv"

# Opening fakeData
with open(fakeData, newline ='') as csvfile:
  fake = csv.reader(csvfile, delimiter =' ')
  for row in fake:
    print(row)
