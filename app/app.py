from flask import Flask, render_template
import numpy as np
from flask_restful import request
import pickle
from predict import Prediction

app = Flask(__name__)

# Initialize model object
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


@app.route('/')
def man():
    return render_template('~/Projects/movie_prediction/genre_prediction_app/templates/home.html')


@app.route('/predict', methods=['POST'])
def post_predict():
    # Predicts top five genres for POSTed test.csv file
    prediction = Prediction(model, vectorizer)
    data = request.form['synopsis']

    # Load the train csv file as a DataFrame
    X = prediction.text_processing(data)
    X = prediction.vectorizer_transform(np.array(X))
    pred = prediction.predict(X)
    return render_template('~/Projects/movie_prediction/genre_prediction_app/templates/prediction.html', data=pred)


if __name__ == '__main__':
    app.run(debug=True)
