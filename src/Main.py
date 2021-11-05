# Modules
import os
import csv
from pathlib import Path

# Constants
DATA_DIRECTORY = Path(os.path.dirname(os.getcwd()) + "/Data/")

# Fake data and real data files 
fakeData = DATA_DIRECTORY / "Fake.csv"
trueData = DATA_DIRECTORY / "True.csv"

# Opening fakeData
with open(fakeData, newline ='') as csvfile:
  fake = csv.reader(csvfile, delimiter = ' ')
  for row in fake:
    print(row)
