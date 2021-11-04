import csv as csv

with open('Fake.csv', newline='') as csvfile:
    fake = list(csv.reader(csvfile)) 

with open('True.csv', newline='') as csvfile:
    true = list(csv.reader(csvfile)) 
