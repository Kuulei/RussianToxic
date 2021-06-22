import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
# from catboost import CatBoostClassifier
from sklearn.svm import SVC


# single flexible function to train different model types with different features
def train_model(model_name, features_name, custom_features=None, input_labels=None):
    EMOTICONS_PUNCTUATION = os.path.abspath(
        'Features\\Emoticons_Exclamation.pkl')
    IDF_Pickle = os.path.abspath('Features\\idf.pkl')
    if input_labels is None:
        train_path = "MultiClass_Train.csv"
        train_set = pd.read_csv(train_path, sep=',', encoding='utf-8')
        all_labels = train_set["Labels"].tolist()
        labels = []
        for i in range(len(all_labels)):
            if 'NORMAL' in list(all_labels)[i]:
                labels.append("NORMAL")
            else:
                labels.append("TOXIC")
    else:
        labels = input_labels
    print("Labels ready")

    if features_name.lower() == "idf":
        idf_path = IDF_Pickle
        features = pd.read_pickle(idf_path)
        print("Features: idf")

    elif features_name.lower() == "emoticons":
        features_path = EMOTICONS_PUNCTUATION
        features = pd.read_pickle(features_path)
        features = features.drop(columns=['Emoticons', 'Question Marks', 'Exclamation Marks', 'Expressive Punctuation'])
        print("Features: emoticons")

    elif features_name.lower() == "punctuation":
        features_path = EMOTICONS_PUNCTUATION
        features = pd.read_pickle(features_path)
        features = features[['Question Marks', 'Exclamation Marks', 'Expressive Punctuation']]
        print("Features: punctuation")

    elif features_name.lower() == "all":
        idf_path = IDF_Pickle
        idf_features_df = pd.read_pickle(idf_path)
        features_path = EMOTICONS_PUNCTUATION
        add_features_df = pd.read_pickle(features_path)
        features = pd.concat([idf_features_df, add_features_df.drop(columns=['Emoticons'])], axis=1)
        print("Features: all")
    else:
        features = custom_features

    if model_name.lower() == "rfm":
        model = RandomForestClassifier(random_state=0)
        print("Model: Random Forest")
        # model = RandomForestClassifier(max_depth=10, random_state=0)1

    elif model_name.lower() == "logreg":
        model = LogisticRegression()
        print("Model: Logistic Regression")

    # support vector machines take too long to train, >24 hours
    # elif model_name.lower() == "svc" or model_name.lower == "support vector":
    #   model = SVC(kernel="linear")
    #  print("Model: Support Vector Classifier")

    elif model_name.lower() == "dtree" or model_name.lower() == "decision tree":
        model = tree.DecisionTreeClassifier()
        print("Model: Decision Tree")

    elif model_name.lower() == "catboost":
        # model = CatBoostClassifier()
        print("Model: Catboost")

    model.fit(features, labels)
    print("Model trained")

    path = f"Models\\BinaryClass_{model_name}_{features_name}.pkl"
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    print("Model saved")
    print("Done")


# train_model(model_name="",features_name = "", custom_features=None, labels=None)
models_list = ["rfm", "logreg", "catboost" , "dtree"]
features_list = ["idf", "punctuation",  "emoticons", "all",]

models_list = ["dtree"]
features_list = ["all"]

# prepare label data
train_path = "MultiClass_Train.csv"
train_set = pd.read_csv(train_path, sep=',', encoding='utf-8')
all_labels = train_set["Labels"].tolist()
labels = []
for i in range(len(all_labels)):
    if 'NORMAL' in list(all_labels)[i]:
        labels.append("NORMAL")
    else:
        labels.append("TOXIC")

# train all
for modeltype in models_list:
    for featuretype in features_list:
        train_model(model_name=modeltype, features_name=featuretype, input_labels=labels)
        print(f"Model trained with algorithm:{modeltype} and features: {featuretype}")

# Enable all code below for training a custom model with selective features
'''print("Reading previous models")
rfm_model = pandas.read_pickle("Models\\BinaryClass_rfm_all.pkl")
logreg_model = pandas.read_pickle("Models\\BinaryClass_logreg_all.pkl")
idf2_features_df = pd.read_pickle('Features\\idf.pkl')
add2_features_df = pd.read_pickle('Features\\Emoticons_Exclamation.pkl')
all_features = pd.concat([idf2_features_df, add2_features_df.drop(columns=['Emoticons'])], axis=1)

def evaluate_features(model, features, model_type):
    feature_list = list(features.columns)
    indexes = []
    importances = []
    counter = 0
    if model_type.lower() == "rfm" or model_type.lower() == "random forest":
        for item in model.feature_importances_:
            if abs(item) > 0.0003:
                indexes.append(counter)
                importances.append(item)
            counter += 1
    elif model_type.lower() == "logreg" or model_type.lower() == "logistic regression":
        for item in list(model.coef_[0]):
            if abs(item) > 1.5:
                indexes.append(counter)
            importances.append(item)
            counter += 1
    usefull_features = []
    for item in indexes:
        usefull_features.append(feature_list[item])

    #print(len(usefull_features))
    #for i in range(len(usefull_features)):
     #   print(usefull_features[i], importances[i])

    return usefull_features, importances

print("Checking feature coefs")
# select the best logreg features
selected_logreg_features, logreg_importances = evaluate_features(model=logreg_model, features=all_features,
                                                           model_type='logreg')
# select the best random forest features
selected_rfm_features, rfm_importances = evaluate_features(model=rfm_model, features=all_features,
                                                           model_type='rfm')
# combine the features and cast them into a set to get rid of duplicates
collective_features = set(selected_rfm_features + selected_logreg_features)
selected_features = all_features[list(collective_features)]
print("Combined features")
# train a new random forest and logistic regression model with the combined set of useful features
#train_model("rfm","selected_collective", custom_features=selected_features, input_labels=None)
#train_model("logreg","selected_collective", custom_features=selected_features, input_labels=None)
train_model("svc","selected_collective", custom_features=selected_features, input_labels=None)'''
