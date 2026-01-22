from flask import Flask, render_template, request
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load model and scaler ONCE
model = joblib.load("model_lr.pkl")
scaler = joblib.load("scaler1.pkl")

# ---------------- Prediction Logic (UNCHANGED) ----------------
def predict_liver_disease(input_dict):
    expected_features = [
        "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
        "Alkaline_Phosphotase", "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase", "Total_Protiens",
        "Albumin", "Albumin_and_Globulin_Ratio"
    ]

    df = pd.DataFrame([input_dict], columns=expected_features)
    X_scaled = scaler.transform(df)

    prob = model.predict_proba(X_scaled)[0][1]
    y_pred = model.predict(X_scaled)[0]

    pred_class = "Liver Disease" if y_pred == 1 else "No Disease"

    if prob < 0.30:
        risk = "Low Risk"
    elif prob < 0.50:
        risk = "Moderate Risk"
    elif prob < 0.70:
        risk = "High Risk"
    else:
        risk = "Severe Risk"

    return pred_class, risk, round(float(prob), 2)

# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "Age": float(request.form["Age"]),
            "Gender": int(request.form["Gender"]),
            "Total_Bilirubin": float(request.form["Total_Bilirubin"]),
            "Direct_Bilirubin": float(request.form["Direct_Bilirubin"]),
            "Alkaline_Phosphotase": float(request.form["Alkaline_Phosphotase"]),
            "Alamine_Aminotransferase": float(request.form["Alamine_Aminotransferase"]),
            "Aspartate_Aminotransferase": float(request.form["Aspartate_Aminotransferase"]),
            "Total_Protiens": float(request.form["Total_Protiens"]),
            "Albumin": float(request.form["Albumin"]),
            "Albumin_and_Globulin_Ratio": float(request.form["Albumin_and_Globulin_Ratio"])
        }

        predicted_class, risk_level, probability = predict_liver_disease(data)

        result = {
            "Predicted_Class": predicted_class,
            "Risk_Level": risk_level,
            "Probability": probability,
            "Age": data["Age"],
            "Gender": "Male" if data["Gender"] == 1 else "Female"
        }

        report_time = datetime.now().strftime("%d %b %Y, %I:%M %p")

        return render_template(
            "result.html",
            result=result,
            data=data,
            report_time=report_time
        )

    except Exception as e:
        return render_template(
            "result.html",
            error=str(e)
        )

if __name__ == "__main__":
    app.run(debug=True)
