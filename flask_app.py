import pandas as pd
from flask import Flask, request, render_template
import pickle
import joblib

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('CKD.pkl', 'rb'))
try:
    scaler = joblib.load('scaler.pkl')
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
    try:
        # Get form data in correct order
        features = [
            float(request.form['pus_cells']),
            float(request.form['red_blood_cells']),
            float(request.form['blood_glucose_random']),
            float(request.form['blood_urea']),
            float(request.form['pedal_edema']),
            float(request.form['anemia']),
            float(request.form['diabetesmillitus']),
            float(request.form['coronory_artery_disease'])
        ]
        
        # Create DataFrame
        df = pd.DataFrame([features], columns='pus_cells', 'red_blood_cells', 'blood_glucose_random', 'blood_urea','pedal_edema', 'anemia', 'diabetesmillitus', 'coronory_artery_disease'])
        
        # Scale and predict
        if scaler:
            df = scaler.transform(df)
        
        prediction = model.predict(df)[0]
        result = 'CKD Detected' if prediction == 0 else 'No CKD Detected'
        
        return render_template('result.html', prediction_text=result)
        
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
