from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.pipeline import Pipeline
from collections import defaultdict
from joblib import dump
import re
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier

# uses tf idf to vectorize documents and a given classifier
# to classify the documents


class TfidfClassifier:
    def __init__(self, classifier=None, path_to_datadir="data/", ngrams=None):
        self.classifier = classifier
        self.tfidf = TfidfVectorizer(ngram_range=(1, 1))
        self.path_to_datadir = path_to_datadir

    def classify(self, cls_pipeline):
        train_dframe = pd.read_csv(self.path_to_datadir+'train.csv', sep="\t", quotechar="|")
        test_dframe = pd.read_csv(self.path_to_datadir+'test.csv', sep="\t", quotechar="|")

        total_frame = pd.concat([train_dframe, test_dframe], axis=0)
        total_frame = total_frame.sample(frac=1).reset_index(drop=True)
        df_len = total_frame.shape[0]
        train_dframe = total_frame[:int(0.7*df_len)]
        test_dframe = total_frame[int(0.7*df_len):]

        train_sentences = [" ".join(item.split(" ")[:100]) for item in train_dframe['text'].tolist()]
        test_sentences = [" ".join(item.split(" ")[:100]) for item in test_dframe['text'].tolist()]

        train_labels = train_dframe['label'].tolist()
        test_labels = test_dframe['label'].tolist()

        cls_pipeline.fit(train_sentences, train_labels)
        plt.rcParams.update({"font.size": 5})
        plot_confusion_matrix(cls_pipeline, test_sentences, test_labels, normalize='true', cmap='Blues',
                              include_values=True, ax=None, xticks_rotation='vertical', values_format=".2f")
        plt.title("Confusion Matrix van het TF-IDF model")
        plt.xlabel("Voorspelling")
        plt.ylabel("Echte Klasse")
        plt.show()
        quit()
        dump(cls_pipeline, 'svm_model.joblib')
        # return classification_report(test_labels, cls_pipeline.predict(test_sentences))
        return accuracy_score(test_labels, cls_pipeline.predict(test_sentences))
# Try a naive bayes classifier?

# f_cls = SVC(class_weight='balanced')

# f_cls = RandomForestClassifier()


classifier_k_val_scores = defaultdict(list)

# pipelines = [Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())]),
#              Pipeline([('tfidf', TfidfVectorizer()), ('clf', RandomForestClassifier())]),
#              Pipeline([('tfidf', TfidfVectorizer()), ('clf', ComplementNB())]),
#              Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())]),
#              Pipeline([('tfidf', TfidfVectorizer()), ('clf', MLPClassifier(hidden_layer_sizes=[256, 512, 256],
#                                                                            verbose=True, early_stopping=True,
#                                                                            learning_rate='adaptive'))]),
# ]

pipelines = [Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC(class_weight='balanced'))]),
             Pipeline([('tfidf', TfidfVectorizer()), ('clf', RandomForestClassifier())]),
             Pipeline([('tfidf', TfidfVectorizer()), ('clf', ComplementNB())]),
             Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])]

pipeline_names = ['SVM', "Random_forest", "Naive Bayes", "Logistic Regression"]

print("---Starting---")
classifier = TfidfClassifier(classifier=None, path_to_datadir="D:/bert_format_data/full/")
classifier.classify(cls_pipeline=pipelines[0])
# for idx in range(len(pipelines)):
#     for i in tqdm(range(2, 19)):
#         classifier_k_val_scores[pipeline_names[idx]].append(classifier.classify(k_val=i, cls_pipeline=pipelines[idx]))
#
# print(classifier_k_val_scores)
#
# for key, val in classifier_k_val_scores.items():
#     plt.plot(range(2, 19), val, label=key)
# plt.plot(range(2, 19), [0.80 for _ in range(2, 19)])
# plt.title("Prestaties van verschillende modellen \n t.o.v. de hoeveelheid klassen")
# plt.xlabel("Aantal klasses")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()
