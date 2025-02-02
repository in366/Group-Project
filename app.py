from flask import Flask, request, jsonify, render_template
import os
from textblob import TextBlob
import google.generativeai as genai

app = Flask(__name__)

# Display homepage form
@app.route('/')
def index():
    return render_template('index.html')

# Handle loan evaluation request
@app.route('/evaluate_loan', methods=['POST'])
def evaluate_loan():
    data = request.get_json()
    
    # Calculate credit score
    credit_score = calculate_credit_score(data)
    # Determine loan approval based on credit score
    decision = "Approved" if credit_score >= 70 else "Denied"
    # Generate risk analysis report
    risk_analysis = get_risk_analysis(credit_score, data)
    
    return jsonify({
        'credit_score': credit_score,
        'decision': decision,
        'risk_analysis': risk_analysis
    })

def calculate_credit_score(data):
    score = 0

    # 1. Employment Status Score (0-20 points)
    employment_scores = {
        'employed': 20,
        'self-employed': 15,
        'retired': 10,
        'unemployed': 0
    }
    score += employment_scores.get(data.get('employmentStatus', ''), 0)
    
    # 2. Income Score (0-30 points) - Based on income-to-loan ratio
    try:
        monthly_income = float(data.get('income', 0))
        loan_amount = float(data.get('loanAmount', 0))
        loan_period = float(data.get('loanPeriod', 0))
    except ValueError:
        monthly_income, loan_amount, loan_period = 0, 0, 0

    if loan_amount > 0 and loan_period > 0:
        income_ratio = (monthly_income * loan_period) / loan_amount
        if income_ratio > 2.0:
            score += 30
        elif income_ratio > 1.5:
            score += 20
        elif income_ratio > 1.0:
            score += 10
        else:
            score -= 5  # High-risk applicant if income-to-loan ratio is too low
    else:
        score -= 5

    # 3. Default History Score (0-30 points)
    if data.get('defaultRecord', '').lower() == 'no':
        score += 30
    else:
        score -= 10  # Deduction for past defaults

    # 4. Loan Purpose Score (0-20 points)
    purpose_scores = {
        'business': 20,
        'house': 18,
        'education': 15,
        'car': 12,
        'personal': 10,
        'other': 5
    }
    score += purpose_scores.get(data.get('loanUsage', ''), 5)
    
    # 5. Loan Period Penalty (Loans exceeding 180 months lose 5 points)
    if loan_period > 180:
        score -= 5

    # 6. Loan Reason Sentiment Analysis (optional) using TextBlob
    if 'loanReason' in data and data['loanReason'].strip() != "":
        score += analyze_loan_reason(data['loanReason'])
    
    # 7. Bank Account Verification (if account number is too short, deduct 5 points)
    if len(data.get('bankAccount', '')) < 10:
        score -= 5
    
    # Ensure credit score stays within 0 to 100
    score = max(0, min(score, 100))
    
    return score

def analyze_loan_reason(reason):
    blob = TextBlob(reason)
    polarity = blob.sentiment.polarity
    # Positive sentiment → +5 points, Negative sentiment → -5 points, Neutral → No effect
    if polarity > 0.3:
        return 5
    elif polarity < -0.3:
        return -5
    return 0

def get_risk_analysis(score, data):
    analysis = []
    
    # General risk classification based on credit score
    if score >= 80:
        analysis.append("Low-risk applicant")
    elif score >= 60:
        analysis.append("Medium-risk applicant")
    else:
        analysis.append("High-risk applicant")
    
    # Additional risk factors
    try:
        monthly_income = float(data.get('income', 0))
        loan_amount = float(data.get('loanAmount', 0))
        loan_period = float(data.get('loanPeriod', 0))
    except ValueError:
        monthly_income, loan_amount, loan_period = 0, 0, 0

    if loan_amount > monthly_income * 24:
        analysis.append("Loan amount is significantly high compared to income")
    if data.get('defaultRecord', '').lower() == 'yes':
        analysis.append("Previous default history is a concern")
    if loan_period > 180:
        analysis.append("Long loan period increases risk")
    
    # Use Generative AI to supplement risk analysis if API key is set
    if os.environ.get("GOOGLE_API_KEY") and data.get('loanReason', '').strip() != "":
        try:
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            prompt = (
                f"Analyze the loan application risk based on the following data:\n"
                f"Credit Score: {score}\n"
                f"Applicant Data: {data}\n"
                "Provide detailed risk analysis and recommendations in English."
            )
            response = genai.generate_text(
                model="models/text-bison-001",
                prompt=prompt
            )
            if hasattr(response, 'result') and response.result:
                analysis.append("AI-generated analysis: " + response.result)
        except Exception as e:
            analysis.append("AI analysis could not generate additional insights.")
    
    return "; ".join(analysis)

if __name__ == '__main__':
    app.run(debug=True, port=5006)