from flask import Flask, render_template, request
import pickle as pkl
import os
import pandas as pd

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load scaler and model
with open(os.path.join(BASE_DIR, "model", "scaler.pkl"), "rb") as f:
    scaler = pkl.load(f)

with open(os.path.join(BASE_DIR, "model", "nb.pkl"), "rb") as f:
    model = pkl.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = ""
    high_risk = False

    if request.method == "POST":
        try:
            Pregnancies = int(request.form["Pregnancies"])
            Glucose = float(request.form["Glucose"])
            BloodPressure = float(request.form["BloodPressure"])
            SkinThickness = float(request.form["SkinThickness"])
            Insulin = float(request.form["Insulin"])
            BMI = float(request.form["BMI"])
            DPF = float(request.form["DiabetesPedigreeFunction"])
            Age = int(request.form["Age"])

            # Validation
            if not (0 <= Pregnancies <= 20):
                raise ValueError("Pregnancies must be between 0 and 20")
            if not (50 <= Glucose <= 300):
                raise ValueError("Glucose must be between 50 and 300")
            if not (30 <= BloodPressure <= 150):
                raise ValueError("Blood Pressure must be between 30 and 150")
            if not (0 <= SkinThickness <= 100):
                raise ValueError("Skin Thickness must be between 0 and 100")
            if not (0 <= Insulin <= 900):
                raise ValueError("Insulin must be between 0 and 900")
            if not (10 <= BMI <= 70):
                raise ValueError("BMI must be between 10 and 70")
            if not (0.0 <= DPF <= 3.0):
                raise ValueError("DPF must be between 0.0 and 3.0")
            if not (1 <= Age <= 120):
                raise ValueError("Age must be between 1 and 120")

            # DataFrame
            input_df = pd.DataFrame([[ 
                Pregnancies, Glucose, BloodPressure,
                SkinThickness, Insulin, BMI, DPF, Age
            ]], columns=[
                "Pregnancies", "Glucose", "BloodPressure",
                "SkinThickness", "Insulin", "BMI",
                "DiabetesPedigreeFunction", "Age"
            ])

            scaled_input = scaler.transform(input_df.values)
            prediction = int(model.predict(scaled_input)[0])

            if prediction == 1:
                prediction_text = "🔴 HIGH RISK: Chances of Diabetes"
                high_risk = True
            else:
                prediction_text = "🟢 LOW RISK: No Diabetes Detected"

        except ValueError as ve:
            prediction_text = f"⚠️ {ve}"
        except Exception:
            prediction_text = "⚠️ Error occurred. Please check inputs."

    return render_template(
        "index.html",
        prediction=prediction_text,
        high_risk=high_risk
    )

if __name__ == "__main__":
    app.run(debug=True)
