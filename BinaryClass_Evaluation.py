import pandas
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn import tree
from catboost import CatBoostClassifier


def get_model_features(algorithm, feature_name, test_set='default'):
    model = pandas.read_pickle(f"Models\\BinaryClass_{algorithm}_"
                               f"{feature_name}.pkl")
    # model = pandas.read_pickle(f"Models\\BinaryClass_{algorithm}
    # _{feature_name}_2")
    if feature_name.lower() == "idf":
        feature_names = pandas.read_pickle('Features\\idf.pkl')

    elif feature_name.lower() == "emoticons":
        feature_names = pandas.read_pickle(
            'Features\\Emoticons_Exclamation.pkl')
        feature_names = feature_names.drop(columns=['Emoticons', 'Question Marks', 'Exclamation Marks', 'Expressive '
                                                                                                        'Punctuation'])
    elif feature_name.lower() == "punctuation":
        feature_names = pandas.read_pickle(
            'Features\\Emoticons_Exclamation.pkl')
        feature_names = feature_names[['Question Marks', 'Exclamation Marks', 'Expressive Punctuation']]

    elif feature_name.lower() == "all":
        feature_names = pandas.read_pickle('Features\\Test_Features.pkl')

    if test_set == 'default':
        features = pandas.read_pickle('Features\\Test_Features.pkl')
    elif test_set == 'new':
        features = pandas.read_pickle('Features\\Test_Features_2.pkl')

    features = features[list(feature_names.columns)]
    return model, features.values


# get TRUE test labels
test_path = "MultiClass_Test.csv"

# use this path instead for the external test set
# test_path = "Full_SingleClass_1.csv"
test_set = pandas.read_csv(test_path, sep=',', encoding='utf-8')

# if testset is from same set as training
if test_path == "MultiClass_Test.csv":
    test_labels = test_set["Labels"].tolist()
    true_labels = []
    for i in range(len(test_labels)):
        if 'NORMAL' in list(test_labels)[i]:
            true_labels.append("NORMAL")
        else:
            true_labels.append("TOXIC")

# if the testset points to the new dataset
elif test_path == "Full_SingleClass_1.csv":
    labels = test_set["toxic"]
    true_labels = []
    for entry in labels:
        if entry == 1 or entry == "1":
            true_labels.append("TOXIC")
        elif entry == 0 or entry == "0":
            true_labels.append("NORMAL")


# single evaluation function for any model, also generates a plot
def eval_model(model, features, test_labels):
    predictions = model.predict(features)
    print("Confusion Matrix: ")
    print(confusion_matrix(test_labels, predictions, labels=["NORMAL", "TOXIC"]))
    print("Classification report: ")
    print(classification_report(test_labels, predictions))
    plot_confusion_matrix(model, features, true_labels)
    plt.show()


# all models
# model_to_eval = ["rfm", "logreg", "catboost", "dtree"]
# features_to_eval = ["emoticons", "idf", "all"]

# Best models for each type
# model_to_eval = ["catboost", "logreg", "rfm", "dtree"]
# features_to_eval = ["all"]

'''#
model_to_eval = ["dtree", "logreg", "rfm", "catboost"]
features_to_eval = ["all"]

# generate batch statistics, change 'default' to 'new' to evaluate with external testset
for model_type in model_to_eval:
    for feature_type in features_to_eval:
        print(model_type,feature_type)
        model_single, feature_single = get_model_features(model_type, feature_type, test_set='default')
        eval_model(model_single, feature_single, true_labels)'''

# evaluate with selected features: disclaimer: serious drop in accuracy

print("Reading previous models")
rfm_model = pandas.read_pickle("Models\\BinaryClass_rfm_all.pkl")
logreg_model = pandas.read_pickle("Models\\BinaryClass_logreg_all.pkl")
idf2_features_df = pandas.read_pickle('Features\\idf.pkl')
add2_features_df = pandas.read_pickle(
    'Features\\Emoticons_Exclamation.pkl')
all_features = pandas.concat([idf2_features_df, add2_features_df.drop(columns=['Emoticons'])], axis=1)


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

    # print(len(usefull_features))
    # for i in range(len(usefull_features)):
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
collective_features = list(set(selected_rfm_features + selected_logreg_features))
print("Combined features")

selected_rfm_model = pandas.read_pickle("Models"
                                        "\\BinaryClass_rfm_selected_collective.pkl")
selected_logreg_model = pandas.read_pickle(
    "Models\\BinaryClass_logreg_selected_collective.pkl")
# selected_svc_model = pandas.read_csv(
# "Models\\BinaryClass_svc_selected_collective.pkl")

selected_features = pandas.read_pickle('Features\\Test_Features.pkl')
selected_features = selected_features[collective_features]

print("Logreg with selected features")
eval_model(selected_logreg_model, selected_features, true_labels)

print("RFM with selected features")
eval_model(selected_rfm_model, selected_features, true_labels)

# print("Support Vector Classifier with selected features")
# eval_model(selected_svc_model, selected_features, test_labels)
