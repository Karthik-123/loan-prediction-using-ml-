from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load('loan_model.pkl')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form
    input_data = (
        int(data['Gender']),
        int(data['Married']),
        int(data['Dependents']),
        int(data['Education']),
        int(data['Self_Employed']),
        float(data['ApplicantIncome']),
        float(data['CoapplicantIncome']),
        float(data['LoanAmount']),
        float(data['Loan_Amount_Term']),
        int(data['Credit_History']),
        int(data['Property_Area'])
    )

    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)

    # Pass the prediction result to the result template
    return render_template("result.html", prediction=int(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
