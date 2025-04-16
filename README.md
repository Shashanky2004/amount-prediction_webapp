# amount-prediction_webapp# Amount-prediction-ml-model
Predicting Loan Amount to be granted using Datasets given.
Introduction
Report Title: An In-depth Analysis of Loan Amount Prediction Models
In the contemporary financial landscape, access to loans plays a pivotal role in the realization of
personal and professional aspirations. Whether it's a first-time homebuyer seeking a mortgage, a
budding entrepreneur in need of startup capital, or an individual looking to finance their education,
the process of securing a loan is a critical facet of financial planning. Lenders are tasked with
assessing the creditworthiness of applicants and determining the optimal loan amount to disburse.
The objective of this report, titled "An In-depth Analysis of Loan Amount Prediction Models," is to
explore and evaluate various machine learning models to predict loan amounts accurately.
Accurate loan amount predictions are fundamental for both borrowers and lenders. For borrowers,
it ensures they receive the financial assistance they need, while for lenders, it minimizes the risk
associated with default.
The report primarily focuses on three distinct regression models for loan amount prediction:
1. Linear Regression: This model seeks to establish a linear relationship between various
applicant attributes and the loan amount. It is a fundamental and widely used approach for
predictive modeling.
2. Polynomial Regression: Polynomial regression introduces nonlinear relationships between the
predictors and the loan amount by including polynomial terms. This approach allows for a more
flexible modeling of complex data patterns.
3. Random Forest Regression: Random forest regression leverages an ensemble of decision
trees to predict loan amounts. This model excels in capturing intricate relationships within the data.
In addition to model selection, this report encompasses critical phases of the data analysis
process, including data preprocessing, feature engineering, and model evaluation. A key focus is
on assessing the accuracy and generalization capabilities of each model through metrics such as
Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2) score.
Ultimately, the objective of this study is to identify the most effective loan amount prediction model
that can be deployed in real-world financial settings. This selection is critical for banks, financial
institutions, and credit agencies to streamline their lending processes and ensure better financial
outcomes for applicants.
The subsequent sections of this report delve into the methodology, results, and discussion,
culminating in a comprehensive recommendation for the ideal loan amount prediction model. The
findings presented in this report aim to contribute to the advancement of predictive modeling in the
finance sector, ultimately benefiting both lenders and borrowers alike.

# Loan Amount Prediction Web Application

This is a web application that predicts loan amounts based on various parameters using machine learning models.

## Features

- Modern and responsive user interface
- Input validation and error handling
- Real-time loan amount prediction
- Support for multiple input parameters

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Amount-prediction-ml-model
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Input Parameters

- Gender
- Marital Status
- Number of Dependents
- Education Level
- Employment Status
- Applicant Income
- Co-applicant Income
- Loan Term
- Credit History
- Property Area

## Model Integration

The application is designed to work with a trained machine learning model saved as `model.pkl`. If the model file is not found, it will use a simple heuristic-based prediction.

## Development

To modify the application:

1. Edit `app.py` for backend logic
2. Modify `templates/index.html` for frontend changes
3. Add new styles in `static/css/style.css`

## License

This project is licensed under the MIT License - see the LICENSE file for details.
