import joblib
import pandas as pd
import gradio as gr


model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")


def predict_loan(gender, married, education, applicant_income, loan_amount, credit_history):
    
    input_dict = {
        'ApplicantIncome': applicant_income,
        'LoanAmount': loan_amount,
        'Credit_History': float(credit_history),
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Married_Yes': 1 if married == 'Yes' else 0,
        'Education_Not Graduate': 1 if education == 'Not Graduate' else 0
    }
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    return "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"


demo = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Radio(["Yes", "No"], label="Married"),
        gr.Radio(["Graduate", "Not Graduate"], label="Education"),
        gr.Number(label="Applicant Income"),
        gr.Number(label="Loan Amount (in thousands)"),
        gr.Radio(["1.0", "0.0"], label="Credit History")
    ],
    outputs=gr.Textbox(label="Loan Decision"),
    title="Loan Approval Predictor",
    description="Enter applicant details to check if the loan will be approved."
)

demo.launch()
