import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('CKD.pkl', 'rb'))
try:
    scaler = pickle.load(open('scalar.pkl', 'rb'))
except:
    scaler = None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Prediction')
def predict_form():
    return render_template('indexnew.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [
        float(request.form['hemogloblin']),
        float(request.form['packed_cell_volume']),
        float(request.form['specific_gravity']),
        float(request.form['red_blood_cell_count']),
        float(request.form['hypertension']),
        float(request.form['diabetesmillitus']),
        float(request.form['albumin']),
        float(request.form['blood_glucose_random'])
    ]
    
    # Create DataFrame
    df = pd.DataFrame([features], columns=['hemogloblin', 'packed_cell_volume', 'specific_gravity', 'red_blood_cell_count','hypertension', 'diabetesmillitus', 'albumin', 'blood_glucose_random']
    )
    
    # Scale and predict
    if scaler:
        df = scaler.transform(df)
    
    prediction = model.predict(df)[0]
    result = 'CKD Detected' if prediction == 0 else 'No CKD Detected'
    
    return render_template('result.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
