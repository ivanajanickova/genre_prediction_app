import numpy as np
import neattext as nt
import neattext.functions as nfx
import pickle


class Prediction():

    def __init__(self, model, vectorizer):
        self.clf = model
        self.vectorizer = vectorizer

    def text_processing(self, data):
        # Explore For Noise
        data.apply(lambda x: nt.TextFrame(x).noise_scan())
        # Explore For Noise
        data.apply(lambda x: nt.TextExtractor(x).extract_stopwords())
        # Explore For Noise
        data.apply(nfx.remove_stopwords)

        X = data.apply(nfx.remove_stopwords)

        return X

    def vectorizer_transform(self, X):
        # Fits TfidVectorized to text
        X = self.vectorizer.transform(X)
        return X

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

    def get_top_three(self, y_labels):
        # Returns the top three genres predicted
        n = 5
        names = self.y.columns
        top_n_labels_idx = np.argsort(-y_labels, axis=1)[:, :n]
        top_n_labels = [names[i] for i in top_n_labels_idx]
        return top_n_labels

    def predict(self, X):
        # Returns the predicted class in an array
        y_pred = self.clf.predict_proba(X)
        top_three = self.get_top_three(y_pred)
        return self.format_output(top_three)
