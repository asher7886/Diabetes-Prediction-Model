from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and preprocessing tools
data_bundle = joblib.load('processed_data.joblib')
scaler = data_bundle['scaler']
features = data_bundle['features']
status_map = data_bundle['status_map']
inv_status_map = {v: k for k, v in status_map.items()}

# Load the best model (Voting Classifier or Random Forest)
model = joblib.load('results/model_Voting_Classifier.joblib')

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        input_data = []
        for feat in features:
            val = request.form.get(feat)
            if feat in ['gender', 'smoking_status', 'family_history']:
                # Basic mapping for categorical inputs
                if feat == 'gender':
                    val = 1 if val.lower() == 'male' else 0
                elif feat == 'smoking_status':
                    mapping = {'never': 2, 'former': 1, 'current': 0}
                    val = mapping.get(val.lower(), 2)
                elif feat == 'family_history':
                    val = 1 if val.lower() == 'yes' else 0
            input_data.append(float(val))
            
        # Scale and predict
        input_scaled = scaler.transform([input_data])
        prediction_idx = model.predict(input_scaled)[0]
        status = inv_status_map[prediction_idx]
        
        # Recommendations
        recommendations = {
            'Non-Diabetic': "Maintain a healthy lifestyle with balanced diet and regular exercise.",
            'Pre-Diabetic': "Reduce sugar intake, increase physical activity, and consult a doctor for a check-up.",
            'Diabetic': "Strictly follow medical advice, monitor glucose levels regularly, and maintain a diabetic-friendly diet."
        }
        
        return render_template('index.html', 
                             prediction=status, 
                             recommendation=recommendations[status],
                             features=features)
    except Exception as e:
        return render_template('index.html', error=str(e), features=features)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
