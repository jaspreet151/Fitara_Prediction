# FITARA Document Classifier

Showcases FITARA classifier trained on 955 raw text documents with over 5 million words. Primary goal is to pipe the input text and to whether the document text is a FITARA Doc. in real time.

Model is trained using both Machine Learning and Deep Learning. The SVC(state vector classifier) model with penalty factor(C=10000) is used to train the machine learning model. RNN is used to train the Deep Learning model.

This is a demo project to elaborate how Machine Learn and Deep Learning Models are deployed on production using Flask API.

## Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

## Project Structure
This project has four major parts:
1.	model.py - This contains code for our Machine Learning and Deep Learning model to predict the text document as FITARA or not.
2.	app.py - This contains Flask APIs that receives text through GUI or API calls, computes the precited value based on our model and returns it.
3.	templates - This folder contains the HTML template to allow user to enter the text and displays the predicted value i.e. True or False.
4.	Model – This folder contains the vectorizer, model pickles for ML model and tokenizer pickle, DL model h5 file.

## Running the project
1.	Ensure that you are in the project home directory. Run app.py using below command to start Flask API
<img width="65" alt="cm1" src="https://user-images.githubusercontent.com/41380402/65391395-60267500-dd86-11e9-8984-9d2cf2245c7f.PNG">

2.	By default, flask will run on port 5000.
 	  Navigate to URL http://localhost:5000
    You should be able to view the homepage as below :
<img width="960" alt="p1" src="https://user-images.githubusercontent.com/41380402/65409145-90145d80-de04-11e9-8bf2-7568c1d0a097.PNG">

3.	Select one of the three options i.e. ML Model, DL Model and Prediction Table.
<img width="742" alt="p2" src="https://user-images.githubusercontent.com/41380402/65391482-4a657f80-dd87-11e9-9fb2-8b42bcd7de89.PNG">

4.  Enter the document text(for DL min. 100 words) and hit Predict.
<img width="960" alt="p3" src="https://user-images.githubusercontent.com/41380402/65391465-18ecb400-dd87-11e9-964c-886e17df7bd1.PNG">
If everything goes well, you should be able to see the prediction on the HTML page!

5.	You can see the logs of every input and prediction made by going to the Prediction table Column.
<img width="336" alt="table1" src="https://user-images.githubusercontent.com/41380402/65391470-2ace5700-dd87-11e9-859a-0086f18bb52a.PNG">
