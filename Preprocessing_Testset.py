import os
import spacy
import numpy
import pandas
import pandas as pd
from Aardberrie2 import sort_dict_key, punctuation_features, punctuation_features_binary, emoji_feature
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the training data from a csv file, change path for different testset
# test_path = "MultiClass_Test.csv"
test_path = "Full_SingleClass_1.csv"
test_set = pandas.read_csv(test_path, sep=',', encoding='utf-8')
if test_path == "MultiClass_Test.csv":
    test_comments = test_set["Comments"].to_list()

elif test_path == "Full_SingleClass_1.csv":
    test_comments = test_set["comment"]


def punctuation_features_binary(comments):
    # create feature vectors for binary of question marks, exclamation marks and combined expressive punctuation
    punct_q = numpy.zeros((len(comments), 1))
    punct_excl = numpy.zeros((len(comments), 1))
    punct_express = numpy.zeros((len(comments), 1))
    express_punct = ['?!', '!!', '!?', '...']
    counter = 0
    for comment in comments:
        if '?' in comment:
            punct_q[counter] = 1
        if '!' in comment:
            punct_excl[counter] = 1
        for combo in express_punct:
            if combo in comment:
                punct_express[counter] = 1
        counter += 1
    return punct_q, punct_excl, punct_express


punct_q, punct_excl, punct_express = punctuation_features_binary(test_comments)
print('Punctuation generated')
# get the emoji list previously generated to use as vectors
emo_features = pd.read_pickle('Features\\Emoticons_Exclamation.pkl')
emo_features = emo_features.drop(columns=['Emoticons', 'Question Marks', 'Exclamation Marks', 'Expressive Punctuation'])
emo = list(emo_features.columns)
emo_vectors = numpy.zeros((len(test_comments), len(emo)))
# update feature vectors for the emo features
for comment, f in list(zip(test_comments, emo_vectors)):
    for term in emo:
        if str(term) in comment:
            term_id = emo.index(term)
            f[term_id] = 1
print('Emoji Generated')
print(emo)


def idf_feature(comments, vocabulary=None):
    # uses a Russian text and creates tf-idf features represented in a dataframe
    nlp = spacy.load("ru_core_news_md")
    train_doc = nlp.pipe(comments)
    result_list = []
    for comment in train_doc:
        new_comment = []
        for token in comment:
            if not token.is_stop and token.is_alpha or token.is_punct:
                new_comment.append(token.lemma_)
        result_list.append(" ".join(new_comment))
    cleaned_doc = pd.DataFrame(result_list, columns=['Comments'])
    if vocabulary is None:
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), min_df=0.0005)
    else:
        vectorizer = TfidfVectorizer(lowercase=True, vocabulary=vocabulary, ngram_range=(1, 1), min_df=0.0005)
    vectorizer.fit(cleaned_doc['Comments'])
    X = vectorizer.transform(cleaned_doc['Comments'])
    tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    return tfidf


# get the wordlist used as features in the first list
previous_idf = pandas.read_pickle('Features\\idf.pkl')
previous_words = list(previous_idf.columns)
# create a df with the idf-features
idf = idf_feature(test_comments, vocabulary=previous_words)
print('idf generated')

# make df with emoji, punctuation features and combine with idf features
df = pd.DataFrame(list(zip(emo_vectors, punct_q, punct_excl, punct_express)),
                  columns=['Emoticons', 'Question Marks', 'Exclamation Marks', 'Expressive Punctuation'])
df[emo] = pd.DataFrame(df['Emoticons'].tolist(), index=df.index)
all_features = pd.concat([idf, df.drop(columns=['Emoticons'])], axis=1)
print(all_features.head())

if test_path == "MultiClass_Test.csv":
    test_features_DONE = os.path.abspath('Features\\Test_Features.pkl')
elif test_path == "Full_SingleClass_1.csv":
    test_features_DONE = os.path.abspath('Features\\Test_Features_2'
                                         '.pkl')
all_features.to_pickle(path=test_features_DONE)
print('Done')
