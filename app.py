import numpy as np
import sql
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from model import SVCModel
import sqlite3
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#conn = sqlite3.connect('database.db')
#print ("Opened database successfully")

#conn.execute('CREATE TABLE table1 (Input TEXT, Prediction TEXT)')
#print ("Table created successfully")
#conn.close()

app = Flask(__name__)

model = SVCModel()

clf_path = 'C:/Users/jaspreetsingh5/Documents/Fitara_pred_flask/models/final_prediction.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

vec_path = 'C:/Users/jaspreetsingh5/Documents/Fitara_pred_flask/models/CountVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)


model.clf1 = load_model(r'C:\Users\jaspreetsingh5\Documents\flask\my_model.h5')

vec_path1 = 'C:/Users/jaspreetsingh5/Documents/Fitara_pred_flask/models/rnntokenizernn.pkl'
with open(vec_path1, 'rb') as f:
    model.vectorizer1 = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
      int_features = request.form.values()
      uq_vectorized = model.vectorizer_transform(int_features)
      prediction = model.predict(uq_vectorized)
      
      if prediction == 0:
            pred_text = 'False'
      else:
            pred_text = 'True'
      output= pred_text

      text1=request.form['text1']
      with sqlite3.connect("database.db") as con:
            cur = con.cursor()
            cur.execute("INSERT INTO table4 (Input,Prediction) VALUES (?,?)",(text1,output) )
            
            con.commit()
      return render_template('index.html', prediction_text='Is the text Fitara? {}'.format(output))

@app.route('/predictrnn')
def home1():
    return render_template('index1.html')
@app.route('/predictrnn1',methods=['POST'])
def predictrnn1():
    
      int_features = request.form.values()
      uq_vectorized = model.vectorizer_transformrnn(int_features)
      prediction = model.predictrnn(uq_vectorized)
      
      
      
      if prediction > 0.50:
            pred_text = 'True'
      else:
            pred_text = 'False'
      output= pred_text

      text1=request.form['text1']
      with sqlite3.connect("database.db") as con:
            cur = con.cursor()
            cur.execute("INSERT INTO table4 (Input,Prediction) VALUES (?,?)",(text1,output) )
            
            con.commit()
      return render_template('index1.html', prediction_text='Is the text Fitara? {}'.format(output))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([list(data.values())])

    output = prediction
    return jsonify(output)

@app.route('/list')
def list():
   con = sqlite3.connect("database.db")
   con.row_factory = sqlite3.Row
   
   cur = con.cursor()
   cur.execute("select * from table2")
   
   rows = cur.fetchall();
   return render_template("list.html",rows = rows)

if __name__ == "__main__":
    app.run(debug=True)
