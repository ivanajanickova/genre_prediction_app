import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import neattext as nt
import neattext.functions as nfx


class NLPModel():

    def __init__(self):
        # Simple NLP classifier
        sgd = SGDClassifier(loss='log')
        self.clf = OneVsRestClassifier(sgd)
        self.vectorizer = TfidfVectorizer()
        self.mlb = MultiLabelBinarizer(sparse_output=True)
        self.y = None

    def encode_classes(self, data):
        # Encode the movie genres as dummy
        data.genres = [row.split(' ') for row in data.genres]
        data = data.join(
            pd.DataFrame.sparse.from_spmatrix(
                self.mlb.fit_transform(data.pop('genres')),
                columns=self.mlb.classes_,
                index=data.index))

        self.y = data.iloc[:, 3:]

        return self.y

    def text_processing(self, data):
        # Explore For Noise
        data.synopsis.apply(lambda x: nt.TextFrame(x).noise_scan())
        # Explore For Noise
        data.synopsis.apply(lambda x: nt.TextExtractor(x).extract_stopwords())
        # Explore For Noise
        data.synopsis.apply(nfx.remove_stopwords)

        X = data.synopsis.apply(nfx.remove_stopwords)

        return X

    def vectorizer_fit(self, X):
        # Fits TfidVectorized to text
        X = self.vectorizer.fit(X)
        return X

    def vectorizer_transform(self, X):
        # Fits TfidVectorized to text
        X = self.vectorizer.transform(X)
        return X

    def train(self, X, y):
        # Trains the classifier
        X = X
        self.clf.fit(X, y)

    def format_output(self, y_labels):
        # Formats the `predicted-genres` column
        for i in range(0, len(y_labels)):
            row = y_labels[i]
            new_row = ''
            if len(row) != 0:
                for j in range(0, 5):
                    new_row = new_row + row[j] + ' '
            y_labels[i] = new_row
        return y_labels

    def get_top_five(self, y_labels):
        # Returns the top five genres predicted
        n = 5
        names = self.y.columns
        top_n_labels_idx = np.argsort(-y_labels, axis=1)[:, :n]
        top_n_labels = [names[i] for i in top_n_labels_idx]
        return top_n_labels

    def predict(self, X, id):
        # Returns the predicted class in an array
        y_pred = self.clf.predict_proba(X)
        top_five = self.get_top_five(y_pred)
        d = {'movie_id': id, 'predicted-genres': self.format_output(top_five)}
        df = pd.DataFrame.from_dict(d, orient='columns')
        df.to_csv('submission.csv', index=False, index_label=False)
