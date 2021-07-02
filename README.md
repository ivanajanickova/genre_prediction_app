# Movie genre predicion 

This app uses synopsis of a movie to predict a movie genre using NLP. It return a three genres predicted for a given synopsis. It was trained using `train.csv` (data/train.csv). It is deployed with aws. 

## NLP model

The model is a multilabel text classification. For model a `SGDClassifier` was used. The training data were processed using `TDIFvectorizer`. 

## Flask REST API 
