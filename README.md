# FITARA Document Classifier

Showcases FITARA classifier trained on 955 raw text documents with over 5 million words. Primary goal is to pipe the input text and to whether the document text is a FITARA Doc. in real time.
Model is trained using both Machine Learning and Deep Learning. The SVC(state vector classifier) model with penalty factor(C=10000) is used to train the machine learning model. RNN is used to train the Deep Learning model.
This is a demo project to elaborate how Machine Learn and Deep Learning Models are deployed on production using Flask API
Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.
Project Structure
This project has four major parts:
1.	model.py - This contains code for our Machine Learning and Deep Learning model to predict the text document as FITARA or not.
2.	app.py - This contains Flask APIs that receives text through GUI or API calls, computes the precited value based on our model and returns it.
3.	templates - This folder contains the HTML template to allow user to enter the text and displays the predicted value i.e. True or False.
4.	Model – This folder contains the vectorizer, model pickles for ML model and tokenizer pickle, DL model h5 file.
