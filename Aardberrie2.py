import csv
import os
import pandas
import pandas as pd
import numpy
import spacy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from catboost import CatBoostClassifier, Pool
from sklearn import tree
from sklearn.svm import SVC

def lists_to_csv(path, labels, texts):
    with open(path, mode='w', encoding='utf-8') as new_doc:
        writer = csv.writer(new_doc, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(labels)):
            writer.writerow([labels[i], texts[i]])
    print("Finished writing csv to: ", path)


def text_from_path(path):
    # returns a textstring from a given path
    with open(path, encoding='utf-8') as f:
        text = ""
        single_line = f.readline()
        while single_line:
            text += single_line
            single_line = f.readline()
    return text


def sort_dict_key(dictionary):
    # sorts a dictionary according to value descending, returns in a list of tuples
    from operator import itemgetter
    sorted_dict = sorted(dictionary.items(), key=itemgetter(1), reverse=True)
    return sorted_dict



def punctuation_features(comments):
    # takes a list of comments and turns it into two feature vectors
    # create empty feature vectors for number of question marks and exclamation marks
    punct_q = numpy.zeros((len(comments), 1))
    punct_ex = numpy.zeros((len(comments), 1))
    counter = 0
    for comment in comments:
        for char in comment:
            if char == '?':
                punct_q[counter] +=1
            elif char == '!':
                punct_ex[counter] += 1
        counter += 1
    return punct_q, punct_ex


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


def emoji_feature(comments):
    import string
    # create a feature list for with all the basic emoticons
    emo_feature = [':D', ';D', ':-D', ':-d', ':d', ';d', ';)', ':)', ':-)', ':p', ';p', ':-p', ':o', ':O', ':-o', ':(',
               ':-(', ':$', ':-$']

    bad_list = ['–', '«', '»', '…', '', '\ufeff', '️', '\u200b', '́', '—', '‍ ']
    # complete the list with emoticon characters in the dataset
    # also check the frequency of the emoji's to decide which ones and how many are worth using as features
    emoji_dic = dict()
    for text in comments:
        for character in text:
            if not character.isalnum() and character not in string.punctuation and character != ' ' and character not in \
                    bad_list and character not in emoji_dic:
                emoji_dic[character] = 1
            elif character in emoji_dic:
                emoji_dic[character] += 1
    sort_emoji_dic = sort_dict_key(emoji_dic)
    emo_feature = []
    print(len(comments))
    for i in range(len(sort_emoji_dic)):
        if sort_emoji_dic[i][1] >= 115:
            emo_feature.append(sort_emoji_dic[i][0])
    return emo_feature


def idf_feature_dummy(comments, write_path):
    # generate a pickle file with tfidf features based on string input and a target path
    from sklearn.feature_extraction.text import TfidfVectorizer
    import spacy
    import pandas as pd
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
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), min_df=0.0005)
    vectorizer.fit(cleaned_doc['Comments'])
    X = vectorizer.transform(cleaned_doc['Comments'])
    tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    tfidf.to_pickle(path=write_path)
    print(tfidf.head())
    print("idf features saved at" + write_path)


def idf_feature(comments, vocabulary = None):
    # takes a Russian text and creates tf-idf features represented in a dataframe
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
        vectorizer = TfidfVectorizer(lowercase=True, vocabulary=vocabulary,ngram_range=(1, 1), min_df=0.0005)
    vectorizer.fit(cleaned_doc['Comments'])
    X = vectorizer.transform(cleaned_doc['Comments'])
    tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    return tfidf


def rfm_goodcoefs(model_path, feature_list_path):
    # takes a model path and path to feature list as input and takes out the features that have a coef above 0
    rfm = pandas.read_pickle(model_path)
    features_df = pd.read_pickle(feature_list_path)
    featureslist = features_df.columns.tolist()
    indexes = []
    counter = 0
    for item in rfm.feature_importances_:
        if item > 0:
            indexes.append(counter)
        counter += 1

    usefull_features = []
    for item in indexes:
        usefull_features.append(featureslist[item])
    return usefull_features


def train_model_multi(model_name, features_name, custom_features=None, input_labels=None):
    # insert model and feature names as strings and trains the multi-label classification model(s)
    if input_labels is None:
        train_path = "MultiClass_Train.csv"
        train_set = pandas.read_csv(train_path, sep=',', encoding='utf-8')
        labels = train_set["Labels"].tolist()

        # for multilabel model
        multi_label = numpy.zeros((len(labels), 4))
        for i in range(len(labels)):
            if 'NORMAL' in labels[i]:
                multi_label[i][0] = 1
            if 'INSULT' in labels[i]:
                multi_label[i][1] = 1
            if 'THREAT' in labels[i]:
                multi_label[i][2] = 1
            if 'OBSCEN' in labels[i]:
                multi_label[i][3] = 1
        print(multi_label[:5], type(multi_label))
        print('Label vectors generated')
    else:
        labels = input_labels
    if features_name.lower() == "idf":
        idf_path = os.path.abspath('Features\\idf.pkl')
        features = pandas.read_pickle(idf_path)
        print("Features: idf")

    elif features_name.lower() == "emoticons":
        features_path = os.path.abspath(
            'Features\\Emoticons_Exclamation.pkl')
        features = pandas.read_pickle(features_path)
        features = features.drop(columns=['Emoticons', 'Question Marks', 'Exclamation Marks', 'Expressive Punctuation'])
        print("Features: emoticons")

    elif features_name.lower() == "punctuation":
        features_path = os.path.abspath(
            'Features\\Emoticons_Exclamation.pkl')
        features = pandas.read_pickle(features_path)
        features = features[['Question Marks', 'Exclamation Marks', 'Expressive Punctuation']]
        print("Features: punctuation")

    elif features_name.lower() == "all":
        idf_path = os.path.abspath('Features\\idf.pkl')
        idf_features_df = pandas.read_pickle(idf_path)
        features_path = os.path.abspath(
            'Features\\Emoticons_Exclamation.pkl')
        add_features_df = pandas.read_pickle(features_path)
        features = pandas.concat([idf_features_df, add_features_df.drop(columns=['Emoticons'])], axis=1)
        print("Features: all")

    else:
        features = custom_features

    if model_name.lower() == "rfm":
        model_type = RandomForestClassifier(random_state=0)
        print("Model: Random Forest")

    elif model_name.lower() == "logreg":
        model_type = LogisticRegression()
        print("Model: Logistic Regression")

    if model_name.lower() == "catboost":
        eval_dataset = Pool(data=features,
                            label=labels)

        model = CatBoostClassifier(loss_function="MultiClass")
        print("Model: Catboost")
        model.fit(eval_dataset)
        path = f"Models\\Multilabel_{model_name}_{features_name}"
        with open(path, 'wb') as file:
            pickle.dump(model, file)
        print("Model saved")
        print("Done")
    else:
        model = MultiOutputClassifier(model_type, n_jobs=-1)
        model.fit(features, labels)
        path = f"Models\\Multilabel_{model_name}_{features_name}"
        with open(path, 'wb') as file:
            pickle.dump(model, file)
            print("Model saved")
            print("Done")