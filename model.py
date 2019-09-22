# ML imports
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer

# from sklearn.ensemble import RandomForestClassifier
import pickle

#from util import plot_roc
# spacy_tok


class SVCModel(object):

    def __init__(self):
        """Simple NLP
        Attributes:
            clf: sklearn classifier model
            vectorizor: TFIDF vectorizer or similar
        """
        self.clf= SVC(C=10000)
        self.vectorizer = CountVectorizer(min_df=8, ngram_range=(1,2))


    def vectorizer_transform(self, X):
        """Transform the text data to a sparse TFIDF matrix
        """
        X_transformed = self.vectorizer.transform(X)
        return X_transformed
    
    def vectorizer_transformrnn(self, X):
        """Transform the text data to a sparse TFIDF matrix
        """
        from keras.preprocessing.sequence import pad_sequences
        seq1 = self.vectorizer1.texts_to_sequences(X)
        X_transformed = pad_sequences(seq1, maxlen=400)
        return X_transformed

    


    def predict(self, X):
        """Returns the predicted class in an array
        """
        y_pred = self.clf.predict(X)
        return y_pred

    def predictrnn(self, X):
        """Returns the predicted class in an array
        """
        y_pred = self.clf1.predict(X)
        return y_pred[0][0]


    
