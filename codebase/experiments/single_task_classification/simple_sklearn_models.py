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
        self.path_to_datadir = path_to_datadir

    def classify(self, cls_pipeline):
        train_dframe = pd.read_csv(self.path_to_datadir+'train.csv')
        test_dframe = pd.read_csv(self.path_to_datadir+'test.csv')

        total_frame = pd.concat([train_dframe, test_dframe], axis=0)
        total_frame = total_frame.sample(frac=1).reset_index(drop=True)
        df_len = total_frame.shape[0]
        train_dframe = total_frame[:int(0.7*df_len)]
        test_dframe = total_frame[int(0.7*df_len):]

        train_sentences = [" ".join(item.split(" ")[:100]) for item in train_dframe['text'].tolist()]
        test_sentences = [" ".join(item.split(" ")[:100]) for item in test_dframe['text'].tolist()]

        train_labels = train_dframe['category'].tolist()
        test_labels = test_dframe['category'].tolist()

        cls_pipeline.fit(train_sentences, train_labels)
        # return classification_report(test_labels, cls_pipeline.predict(test_sentences))
        return classification_report(test_labels, cls_pipeline.predict(test_sentences))


pipeline = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 3))), ('clf', LinearSVC(class_weight='balanced'))])


print("---Starting---")
classifier = TfidfClassifier(classifier=None, path_to_datadir="../.data/enron/")
print(classifier.classify(cls_pipeline=pipeline))
print("---Finished---")
