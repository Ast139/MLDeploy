from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as np
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

# import lasso regression and Standard Scaler picker
lasso=pickle.load(open('Model/laasocv.pkl','rb'))
ss=pickle.load(open('Model/ss.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predictdata():
    if request.method=='POST':
        Temperature=float(request.form.get("Temperature"))
        RH=float(request.form.get("RH"))
        Ws=float(request.form.get("Ws"))
        Rain=float(request.form.get("Rain"))
        FFMC=float(request.form.get("FFMC"))
        DMC= float(request.form.get("DMC"))
        ISI=float(request.form.get("ISI"))
        Classes=float(request.form.get("Classes"))
        Region=float(request.form.get("Region"))

        X=ss.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=lasso.predict(X)
        return render_template('entry.html',results=result[0])

    else:
        return render_template('entry.html')

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)