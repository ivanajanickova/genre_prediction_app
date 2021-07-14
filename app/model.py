from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import neattext as nt
import neattext.functions as nfx
import pickle


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

    def train(self, filepath):
        # Trains the classifier
        df = pd.read_csv(filepath)
        X = self.text_processing(df)
        self.vectorizer.fit(X)
        X = self.vectorizer.transform(X)
        y = self.encode_classes(df)
        self.clf.fit(X, y)


model = NLPModel()
model.train('~/Projects/movie_prediction/genre_prediction_app/data/train.csv')
pickle.dump(model.clf, open('model.pkl', 'wb'))
pickle.dump(model.y, open('classes.pkl', 'wb'))
pickle.dump(model.vectorizer, open('vectorizer.pkl', 'wb'))
