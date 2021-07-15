# Movie genre predicion 

This app uses summaries of a movie to predict a movie genre using NLP. It returns a three genres predicted for a given summary. It was trained using `train.csv` (data/train.csv). It is deployed with heroku and uses Flask framework. *The App will be soon deployed on Heroku* The app can be found on: https://genre-pred.herokuapp.com/ 

### `model.py`

The NLP model is a multilabel text classification. For model a `SGDClassifier` was used. The training data were processed using `TDIFvectorizer`. Once trained, the classifier & and vectorizer objects are serialized for further use in prediction.  

### `predict.py`

The class Prediction takes the user's input, transforms it and with the use of serialized classifier and vectorizer predicts the three genres for a given movie summary. 

### `app.py`

Renders a homescreen form `home.html` template. The inserted movie summary for prediction is sent as a POST request to a server. 
