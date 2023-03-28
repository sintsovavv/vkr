import flask
from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression

app = flask.Flask(__name__, template_folder = 'templates')

model_1 = pickle.load(open('model_for_mod.pkl', 'rb'))
model_2 = pickle.load(open('model_for_str.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('mail.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model_1.predict(final_features)
    output_1 = round(prediction[0], 2)
    prediction_2 = model_2.predict(final_features)
    output_2 = round(prediction_2[0], 2) 
    return render_template('mail.html', output1 = output_1, output2 = output_2)

app.run()