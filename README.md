# RussianToxic
Documentation for the RuToxic Project:

The goal of this project was to experiment with a couple of different algorithms and features for the detection of toxicity in Russian comments.
The primary dataset can be found at:
https://www.kaggle.com/alexandersemiletov/toxic-russian-comments

The secondary dataset was used for additional evaluation and can be found at:
https://www.kaggle.com/blackmoon/russian-language-toxic-comments

Before training the models and preparing the features, you should download the datasets and name them appropriately while also creating two empty directories for the "Features" and "Models"`.

You should run these files in the correct order, like this:
TextToCsv> All_Features> Train_BinaryClass> Preprocessing_Testset>BinaryClass_Evaluation

Quick overview of the different files:
Aardberrie2 : Contains a collection of some functions that are called in the other files. This document allows functions to be reused. There is no reason or need to run file.
TextToCsv: Run this script first to turn the text file into a csv file ready for preprocessing using Regular Expressions. This step also divides the dataset into 90% training data and 10% test data.

TFIDF: Prepares the tf-idf feature vectors and saves them in the Features directory. NOTE: no longer present in the files. Instead use All_Features.
Add_Features: Creates feature vectors for punctuation (?, ! and 'expressive punctuation') in addition to the emoticons. Also saves this in the Feature directory. NOTE: no longer present in the files.Instead use All_Features.
All_Features: This script combines the TFIDF and Add_Features files and creates all feature files for the training set.

Train_BinaryClass: trains models with given features for binary classification
Train_MultiLabel: trains model with given features for multi-label classification

Preprocessing_Testset: creates feature vectors for the testset based on the training set

BinaryClass_Evaluation: presents the test results with a confusion matrix, classification report and plot of the confusion matrix, only for binary classification models
MultiLabel_Evaluation: presents the test results with manually calculated Accuracy, Precision, Recall, F-score and confusion matrix for each label, only for multi-label classification

Feature_Evaluation: shows the (most important) features of random forest or logistic regression models for binary classification based on coef or feature importance.


