from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('heart_disease_pred.joblib')

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = {
            'slope_of_peak_exercise_st_segment': int(request.form['slope_of_peak_exercise_st_segment']),
            'thal': request.form['thal'],
            'resting_blood_pressure': float(request.form['resting_blood_pressure']),
            'chest_pain_type': int(request.form['chest_pain_type']),
            'num_major_vessels': int(request.form['num_major_vessels']),
            'fasting_blood_sugar_gt_120_mg_per_dl': int(request.form['fasting_blood_sugar_gt_120_mg_per_dl']),
            'resting_ekg_results': int(request.form['resting_ekg_results']),
            'serum_cholesterol_mg_per_dl': float(request.form['serum_cholesterol_mg_per_dl']),
            'oldpeak_eq_st_depression': float(request.form['oldpeak_eq_st_depression']),
            'sex': int(request.form['sex']),
            'age': int(request.form['age']),
            'max_heart_rate_achieved': float(request.form['max_heart_rate_achieved']),
            'exercise_induced_angina': int(request.form['exercise_induced_angina'])
        }

        df = pd.DataFrame([features])
        prediction = model.predict(df)[0]

        if prediction == 1:
            result_text = "⚠️ Heart Disease Prediction: Positive"
            result_class = "bg-red-50 text-red-700 border-red-200"
        else:
            result_text = "Heart Disease Prediction: Negative"
            result_class = "bg-emerald-50 text-emerald-700 border-emerald-200"

        return render_template('index.html', prediction_text=result_text, result_class=result_class)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error occurred: {str(e)}", result_class="bg-amber-50 text-amber-700 border-amber-200")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
