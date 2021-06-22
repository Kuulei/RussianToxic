import os
import re
from Aardberrie2 import text_from_path, lists_to_csv

path = 'Full_MultiClass.txt'

# read file and save as text
text = text_from_path(path)

# using regex to extract data from textfile
label_pattern = "(?:__label__\w+,*)+"
labels = [item.replace("__label__", "") for item in re.findall(label_pattern, text)]

text_pattern = "(?<=[A-Z]\s)(.+)$"
text_list = re.findall(text_pattern, text, re.MULTILINE)

# test to write a single csv with the data, used as start for the function
"""
csv_path = "Full_MultiClass.csv"
with open(csv_path, mode='w', encoding='utf-8') as new_doc:
    writer = csv.writer(new_doc, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(labels)):
        writer.writerow([labels[i], text_list[i]])"""

# split the data into test and training
train_data = ["Comments"]
train_labels = ["Labels"]

test_data = ["Comments"]
test_labels = ["Labels"]

# every third comment and label is set to test while the other 9/10's are for training
counter = 0
for i in range(len(labels)):
    if counter == 9:
        test_data.append(text_list[i])
        test_labels.append(labels[i])
        counter = 0
    else:
        train_data.append(text_list[i])
        train_labels.append(labels[i])
        counter += 1

print("Length of full data: ", len(labels))
print("Length of test data: ", len(test_data))
print("Length of training data: ", len(train_data))

train_path = "MultiClass_Train.csv"
test_path = "MultiClass_Test.csv"

lists_to_csv(train_path, train_labels, train_data)
lists_to_csv(test_path, test_labels, test_data)
