import pandas

# emoticon evaluation => load the model and feature list to extract the columns
emo_model = pandas.read_pickle("Models\\BinaryClass_rfm_emoticons.pkl")
# emo_model = pandas.read_pickle("Models\\BinaryClass_logreg_emoticons")
emo_features = pandas.read_pickle('Features\\Emoticons_Exclamation.pkl')
emo_features = emo_features.drop(columns=['Emoticons', 'Question Marks', 'Exclamation Marks', 'Expressive Punctuation'])

# idf evaluation
idf_model = pandas.read_pickle("Models\\BinaryClass_rfm_idf.pkl")
# idf_model = pandas.read_pickle("Models\\BinaryClass_logreg_idf")
idf_features = pandas.read_pickle('Features\\idf.pkl')

# punctuation evaluation
punct_model = pandas.read_pickle("Models\\BinaryClass_rfm_punctuation.pkl")
# punct_model = pandas.read_pickle("Models
# "\\BinaryClass_logreg_punctuation.pkl")
punct_features = pandas.read_pickle('Features\\Emoticons_Exclamation.pkl')
punct_features = punct_features[['Question Marks', 'Exclamation Marks', 'Expressive Punctuation']]

# all feature evaluation
all_model = pandas.read_pickle("Models\\BinaryClass_rfm_all.pkl")
# all_model = pandas.read_pickle("Models\\BinaryClass_logreg_all")

idf2_features_df = pandas.read_pickle('Features\\idf.pkl')
add2_features_df = pandas.read_pickle(
    'Features\\Emoticons_Exclamation.pkl')
all_features = pandas.concat([idf2_features_df, add2_features_df.drop(columns=['Emoticons'])], axis=1)
'''
for label, coefs, intercept in zip(all_model.classes_, all_model.coef_, all_model.intercept_):
    print(label)
    counter = 0
    for t,c in zip(all_features, coefs):
        print(t,c)
        counter +=1
    print(counter)
    print("INTERCEPT:" + str(intercept))
    print()'''


def evaluate_features(model, features, model_type):
    feature_list = list(features.columns)
    indexes = []
    importances = []
    counter = 0
    if model_type.lower() == "rfm" or model_type.lower() == "random forest":
        for item in model.feature_importances_:
            if abs(item) > 0.00003:
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

    print("Number of total features: ", len(features.columns))
    print("Number of useful features: ",len(usefull_features))
    for i in range(len(usefull_features)):
        print(usefull_features[i], importances[i])

    return usefull_features, importances



# selected_emofeatures, emo_importances = evaluate_features(model=emo_model, features=emo_features)
# selected_idf_features, idf_importances = evaluate_features(model=idf_model, features=idf_features)
# selected_punct_features, punct_importances = evaluate_features(model=punct_model, features=punct_features)
selected_all_features, all_importances = evaluate_features(model=all_model, features=all_features,
                                                           model_type='random forest')

# individual_features = selected_idf_features + selected_punct_features+ selected_emofeatures
# individual_importances = idf_importances + punct_importances + emo_importances

# print("Total length of individual features: ", len(individual_features))
# print("Length of combined features: ", len(selected_all_features))

'''
# check which features are considered most important
print("Individual models: ")
print(sorted(zip(individual_importances, individual_features), reverse=True)[:200])
print("\n\n")
print("Combined model: ")
print(sorted(zip(all_importances, selected_all_features), reverse=True)[:200])


# Check which features were gained and lost comparing individual feature models and the combined model

lost_count = 0
for i in range(len(individual_features)):
    if individual_features[i] not in selected_all_features:
        lost_count +=1
        print("Lost the feature with importance: ", individual_features[i], individual_importances[i])
print("lostcount: ", lost_count)

gained_count = 0
for i in range(len(selected_all_features)):
    if selected_all_features[i] not in individual_features:
        print("gained features with importance: ", selected_all_features[i], all_importances[i])
        gained_count += 1
print("gainedcount: ", gained_count)'''
