#
# Model definition, training and testing for Naive Bayes
#

import numpy as np
import pandas as pd
import time 
import joblib
import nltk
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import CategoricalNB, MultinomialNB
from sklearn.metrics import accuracy_score, auc, classification_report, roc_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords

LOAD_MODEL = False

DATASET_PATH = '../Datasets/combined_data.csv'
TEXT_LABEL = 'text'
SPAM_LABEL = 'label'

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english')).union({"escapenumber"})

# For stemming
stemmer = EnglishStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc) if w not in STOP_WORDS)

# Text dataset processing
def load_data(filename, test_size=0.3):
    df: pd.DataFrame = pd.read_csv(filename)
    return train_test_split(df[TEXT_LABEL], df[SPAM_LABEL], test_size=test_size)

class NBClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, alpha=0.1, use_idf = True, analyzer = stemmed_words, chi_portion = 0.5):
        self.alpha = alpha
        self.use_idf = use_idf
        self.analyzer = analyzer
        self.chi_portion = chi_portion

    def set_params(self, **params) -> BaseEstimator:
        for parameter, value in params.items():
            setattr(self, parameter, value)

        self.model = MultinomialNB(alpha=self.alpha)
        self.count_vectorizer = CountVectorizer( analyzer = self.analyzer )
        self.tf_transformer = TfidfTransformer(use_idf=self.use_idf)
        self.chi2 = None

        return self
    
    def get_params(self, deep: bool = True) -> dict:
        return super().get_params(deep)

    def _fit_preprocess(self, X, y):
        # Init sub-estimators
        self.model = MultinomialNB(alpha=self.alpha)
        self.count_vectorizer = CountVectorizer( analyzer = analyzer )
        self.tf_transformer = TfidfTransformer(use_idf=self.use_idf)

        X_train_tf = self.count_vectorizer.fit_transform(X)
        X_train_tf = self.tf_transformer.fit_transform(X_train_tf)
        self.chi2 = SelectKBest(score_func=chi2, k=round(X_train_tf.shape[1]*self.chi_portion))
        X_train_tf = self.chi2.fit_transform(X_train_tf, y)
        return X_train_tf

    def _preprocess(self, X):
        X_test_tf = self.count_vectorizer.transform(X)
        X_test_tf = self.tf_transformer.transform(X_test_tf)
        X_test_tf = self.chi2.transform(X_test_tf)
        return X_test_tf
    
    def fit(self, X_train, y_train):
        self.classes_, y_train = np.unique(y_train, return_inverse=True)
        X_train_tf = self._fit_preprocess(X_train, y_train)
        self.model = self.model.fit(X_train_tf, y_train)
        return self

    def predict(self, X_test):
        X_test_tf = self._preprocess(X_test)
        return self.model.predict(X_test_tf)
    
    def predict_proba(self, X_test):
        X_test_tf = self._preprocess(X_test)
        return self.model.predict_proba(X_test_tf)
    

if __name__ == "__main__":
    # Preprocess and split dataset
    X_train, X_test, y_train, y_test = load_data(DATASET_PATH, 0.3)

    # Train model
    if not LOAD_MODEL:
        param_grid = {
            'alpha': [0.1, 0.5, 1.0],
            'use_idf': [True, False],
            'chi_portion': [0.5, 0.7, 1.0]
        }
        clf = GridSearchCV(estimator=NBClassifier(), param_grid=param_grid, scoring='f1', n_jobs=5, verbose=3, error_score='raise')
        clf.fit(X_train, y_train)
        # Save to disk
        model = clf.best_estimator_
        print(f"\n=== Best model's params:\n {clf.best_params_}\n")
        joblib.dump(model, 'naive_bayes.sav')
    else:
        model = joblib.load('naive_bayes.sav')
        print(f"\n=== Loaded model's params:\n {model.get_params()}\n")

    # Test model
    y_pred = model.predict(X_test)
    print(f"Classification report: \n{classification_report(y_test, y_pred)}")
    
    # Plot ROC curve

    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
