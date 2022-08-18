#import relevant libraries for flask, html rendering and loading the ML model
from flask import Flask,request,url_for,redirect,render_template
import pickle
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load(open('Model.pkl','rb'))
scale = joblib.load(open('Scale.pkl','rb'))

@app.route("/")
def landingPage():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    pregnancies = request.form['1']
    glucose = request.form['2']
    bloodPressure = request.form['3']
    skinThickness = request.form['4']
    insulin = request.form['5']
    bmi = request.form['6']
    dpf = request.form['7']
    age = request.form['8']   

    rowDF = pd.DataFrame([pd.Series([pregnancies,glucose,bloodPressure,skinThickness,insulin,bmi,dpf,age])])
    rowDF_new = pd.DataFrame(scale.transform(rowDF))
    print(rowDF_new)

    #model predicton
    prediction = model.predict_proba(rowDF_new)
    print('The predicted value is = :',prediction[0][1])

    if prediction[0][1] >= 0.5:
        valPred = round(prediction[0][1],3)
        print(f"The Round val {valPred*100}%")
        return render_template('result.html',pred=f'You have a chance of having diabetes.\n\nProbability of you being a diabetic is {valPred*100:.2f}%.\n\nAdvice : Exercise Regularly')
    else:
        valPred = round(prediction[0][1],3)
        return render_template('result.html',pred=f'Congratulations!!!, You are in a Safe Zone.\n\n Probability of you being a diabetic is only {valPred*100:.2f}%.\n\n Advice : Exercise Regularly and maintain like this..!')

if __name__ == '__main__':
    app.run(debug=True)