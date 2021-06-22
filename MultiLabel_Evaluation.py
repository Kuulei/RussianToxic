import numpy
import pandas
from sklearn.metrics import confusion_matrix

model = pandas.read_pickle(f"Models\\Multilabel_rfm_all.pkl")
# model = pandas.read_pickle(f"Models\\Multilabel_logreg_all.pkl")

features = pandas.read_pickle('Features\\Test_Features.pkl').values

train_path = "MultiClass_Test.csv"
train_set = pandas.read_csv(train_path, sep=',', encoding='utf-8')
labels = train_set["Labels"].tolist()

test_normal = []
test_insult = []
test_threat = []
test_obscene = []
# for multilabel model
multi_label = numpy.zeros((len(labels), 4))
for i in range(len(labels)):
    if 'NORMAL' in labels[i]:
        multi_label[i][0] = 1
        test_normal.append(1)
    else:
        test_normal.append(0)

    if 'INSULT' in labels[i]:
        multi_label[i][1] = 1
        test_insult.append(1)
    else:
        test_insult.append(0)

    if 'THREAT' in labels[i]:
        multi_label[i][2] = 1
        test_threat.append(1)
    else:
        test_threat.append(0)

    if 'OBSCEN' in labels[i]:
        multi_label[i][3] = 1
        test_obscene.append(1)
    else:
        test_obscene.append(0)
print(multi_label[:5], type(multi_label))
print('Label vectors generated')

'''
predictions = model.predict(features)
for prediction in predictions[:20]:
    print(prediction, type(prediction))
print(len(predictions))
print(len(predictions[0]))
for i in range(10):
    if predictions[0][i].any == multi_label[i].any:
        print("TRUE", predictions[i])
    else:
        print("False:\nPredicted: ", predictions, "\nActual result:", multi_label[i])'''

testframe = pandas.DataFrame(model.predict(features))
# print(testframe.head)

normal = testframe.iloc[:, 0]
insult = testframe.iloc[:, 1]
threat = testframe.iloc[:, 2]
obscene = testframe.iloc[:, 3]

normal_pos = 0
normal_true_pos = 0
normal_false_negative = 0

insult_pos = 0
insult_true_pos = 0
insult_false_negative = 0

threat_pos = 0
threat_true_pos = 0
threat_false_negative = 0

obscene_pos = 0
obscene_true_pos = 0
obscene_false_negative = 0

correct_count = 0
partially_correct = 0
wrong_count = 0
toxic_predicted = 0

for i in range(len(labels)):
    if normal[i] == 1:
        normal_pos += 1
        if normal[i] == multi_label[i][0]:
            normal_true_pos += 1
    elif normal[i] == 0 and multi_label[i][0] == 1:
        normal_false_negative += 1

    if insult[i] == 1:
        insult_pos += 1
        if multi_label[i][1] == 1:
            insult_true_pos += 1
    elif insult[i] == 0 and multi_label[i][1] == 1:
        insult_false_negative += 1
    if threat[i] == 1:
        threat_pos += 1
        if multi_label[i][2] == 1:
            threat_true_pos += 1
    elif threat[i] == 0 and multi_label[i][2] == 1:
        threat_false_negative += 1

    if obscene[i] == 1:
        obscene_pos += 1
        if multi_label[i][3] == 1:
            obscene_true_pos += 1
    elif obscene[i] == 0 and multi_label[i][3] == 1:
        obscene_false_negative += 1

normal_precision = normal_true_pos / (normal_true_pos + (normal_pos - normal_true_pos))
normal_recall = normal_true_pos / (normal_true_pos + normal_false_negative)
normal_Fscore = (1 + 1) * ((normal_precision) * normal_recall) / (normal_precision + normal_recall)

print("Normal:")
print("Precision: ", normal_precision)
print("Recall: ", normal_recall)
print("F-score: ", normal_Fscore)
normal_confusion_matrix = confusion_matrix(normal, test_normal)
print(normal_confusion_matrix)

insult_precision = insult_true_pos / (insult_true_pos + (insult_pos - insult_true_pos))
insult_recall = insult_true_pos / (insult_true_pos + insult_false_negative)
insult_Fscore = (1 + 1) * ((insult_precision) * insult_recall) / (insult_precision + normal_recall)

print("Insult:")
print("Precision: ", insult_precision)
print("Recall: ", insult_recall)
print("F-score: ", insult_Fscore)
insult_confusion_matrix = confusion_matrix(insult, test_insult)
print(insult_confusion_matrix)

threat_precision = threat_true_pos / (threat_true_pos + (threat_pos - threat_true_pos))
threat_recall = threat_true_pos / (threat_true_pos + threat_false_negative)
threat_Fscore = (1 + 1) * (threat_precision * threat_recall) / (threat_precision + threat_recall)

print("threat:")
print("Precision: ", threat_precision)
print("Recall: ", threat_recall)
print("F-score: ", threat_Fscore)

threat_confusion_matrix = confusion_matrix(threat, test_threat)
print(threat_confusion_matrix)

# generate statistics for
obscene_precision = obscene_true_pos / (obscene_true_pos + (obscene_pos - obscene_true_pos))
obscene_recall = obscene_true_pos / (obscene_true_pos + obscene_false_negative)
obscene_Fscore = (1 + 1) * (obscene_precision * obscene_recall) / (obscene_precision + obscene_recall)

print("obscene:")
print("Precision: ", obscene_precision)
print("Recall: ", obscene_recall)
print("F-score: ", obscene_Fscore)

obscene_confusion_matrix = confusion_matrix(obscene, test_obscene)
print(obscene_confusion_matrix)

for i in range(len(labels)):
    # print("True value: ")
    # print(multi_label[i])
    # print("Prediction: ")
    # print(testframe.values[i])
    if multi_label[i][0] == normal[i] \
            and multi_label[i][1] == insult[i] \
            and multi_label[i][2] == threat[i] \
            and multi_label[i][3] == obscene[i]:
        correct_count += 1
        # print("Correct Prediction")

    elif multi_label[i][0] == normal[i]:
        partially_correct += 1
        # print("Partially correct")

    else:
        wrong_count += 1
        # print("Wrong  Prediction")

    if normal[i] == 0:
        toxic_predicted += 1

print("Completely correct: ", correct_count)
print("Correct toxicity, wrong kind: ", partially_correct)
print("Wrong predictions: ", wrong_count)
print("Toxic predicted: ", toxic_predicted)

print("Accuracy: ", correct_count / len(labels))
print("Accuracy of Toxicity: ", (correct_count + partially_correct) / len(labels))

# print(confusion_matrix(multi_label, predictions))
