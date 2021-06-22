import numpy
import pandas
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from catboost import CatBoostClassifier, Pool

# load the training set to get the labels
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


def train_model_multi(model_name, features_name, custom_features=None, input_labels=None):
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
        path = f"Models\\Multilabel_{model_name}_{features_name}.pkl"
        with open(path, 'wb') as file:
            pickle.dump(model, file)
        print("Model saved")
        print("Done")
    else:
        model = MultiOutputClassifier(model_type, n_jobs=-1)
        model.fit(features, labels)
        path = f"Models\\Multilabel_{model_name}_{features_name}.pkl"
        with open(path, 'wb') as file:
            pickle.dump(model, file)
            print("Model saved")
            print("Done")


models_list = ["rfm", "logreg"]
features_list = ["idf", "punctuation",  "emoticons", "all",]


# train all
for single_model in models_list:
    for featuretype in features_list:
        train_model_multi(model_name=single_model, features_name=featuretype, input_labels=multi_label)
        print(f"Model trained with algorithm:{single_model} and features: {featuretype}")
