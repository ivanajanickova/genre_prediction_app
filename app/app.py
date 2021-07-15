from flask import Flask, render_template
import pandas as pd
from flask_restful import request
import pickle
from predict import Prediction

app = Flask(__name__, template_folder='templates')

# Initialize model object
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
y = pickle.load(open('classes.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Predicts top three genres for a provided synopsis
    prediction = Prediction(model, vectorizer, y)
    data = request.form['synopsis']
    # transform the text to data frame - input for the vectorizer
    data = pd.DataFrame({'synopsis': data}, index=[0])
    X = prediction.text_processing(data)
    X = prediction.vectorizer_transform(X)
    pred = prediction.predict(X)
    return render_template('home.html', prediction=pred)


if __name__ == '__main__':
    app.run(debug=True)
