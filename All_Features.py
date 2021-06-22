import os
import string
import spacy
import numpy
import pandas
import pandas as pd
from Aardberrie2 import sort_dict_key, punctuation_features, punctuation_features_binary, emoji_feature, idf_feature
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the training data from a csv file
train_path = "MultiClass_Train.csv"
train_set = pandas.read_csv(train_path, sep=',', encoding='utf-8')
train_set_specific = train_set[["Comments", "Labels"]]

train_comments = train_set["Comments"].to_list()
train_labels = train_set["Labels"].to_list()

# generate punctuation features
print("Generating punctuation features")
punct_q, punct_excl, punct_express = punctuation_features_binary(train_comments)
emo_feature = emoji_feature(train_comments)
print("Finished generating punctuation features")

# save the selection of emoticons in a dataframe
print("Generating emoticon features")
all_emoji = pandas.DataFrame(emo_feature)
all_emoji_path = 'Features\\all_emoticons.pkl'
all_emoji.to_pickle(path=all_emoji_path)
print("Finished generating emoticon features")

# create feature vectors for emo's
emo_vectors = numpy.zeros((len(train_comments), len(emo_feature)))
# update feature vectors for the emo features
for comment, f in list(zip(train_comments, emo_vectors)):
    for term in emo_feature:
        if term in comment:
            term_id = emo_feature.index(term)
            f[term_id] = 1

print(emo_vectors.shape)
print(emo_vectors[:50])

# save the punctuation and emoticon features together
df = pd.DataFrame(list(zip(emo_vectors, punct_q, punct_excl, punct_express)), columns=['Emoticons', 'Question Marks',
                                                                                       'Exclamation Marks',
                                                                                       'Expressive Punctuation'])
df[emo_feature] = pd.DataFrame(df['Emoticons'].tolist(), index=df.index)
features_path = os.path.abspath('Features\\Emoticons_Exclamation.pkl')
df.to_pickle(path=features_path)

# generate idf features
print("Generating idf features")
idf_path = os.path.abspath('Features\\idf.pkl')
tfidf = idf_feature(train_comments)
print(type(tfidf), tfidf.head, tfidf.shape)
tfidf.to_pickle(path=idf_path)
print("Finished generating idf features")

print("Done")
