import pickle as pkl
import os
import pandas as pd

# Load scaler and model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pkl.load(f)

with open(os.path.join(BASE_DIR, "nb.pkl"), "rb") as f:
    model = pkl.load(f)

def predict_diabetes(
    Pregnancies,
    Glucose,
    BloodPressure,
    SkinThickness,
    Insulin,
    BMI,
    DiabetesPedigreeFunction,
    Age
):
    input_df = pd.DataFrame([[
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
    ]], columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ])

    scaled_input = scaler.transform(input_df)
    prediction = int(model.predict(scaled_input)[0])

    if prediction == 1:
        print("HIGH RISK: You have high chances of Diabetes.")
    else:
        print("LOW RISK: You have low chances of Diabetes.")

    return prediction


# Test run
if __name__ == "__main__":
    predict_diabetes(
        Pregnancies=2,
        Glucose=190,
        BloodPressure=120,
        SkinThickness=37,
        Insulin=85,
        BMI=37.5,
        DiabetesPedigreeFunction=0.627,
        Age=45
    )
