# Churn-Prediction
ChurnQuest: Telecom Customer Churn Prediction & Analysis
This project analyzes telecom customer data to understand the key drivers of churn and deploys a machine learning model as an interactive web app to predict at-risk customers.

Key Insights & Drivers of Churn
Analysis of the data revealed that the following factors are the most significant predictors of a customer leaving:

High Usage Costs: Total Day Charge is the single most important factor. Customers with high daily usage bills are more likely to churn.

Poor Customer Service: The Number of Customer Service Calls is the second most important predictor. Customers who have to call support multiple times are at a very high risk of leaving.

International Plan: Customers with an International Plan have a higher tendency to churn, suggesting potential issues with the plan's value or cost.

The Solution: A Predictive Model & App
A RandomForestClassifier model was trained to predict churn likelihood. It achieved 95.9% accuracy on the test data and successfully identified 77% of all actual churners (recall).

This model was deployed using Streamlit into an interactive web application where users can input customer details and receive an instant churn probability score, enabling proactive retention efforts.

How to Run This Project
Prerequisites
Python 3.7+

The train.csv dataset in the root directory.

Installation & Setup
Clone the repository:

git clone [your-repository-link]
cd [repository-name]

Install the required libraries:

pip install streamlit pandas scikit-learn joblib

Running the Application
Train the Model (Run this once):
This script trains the model and saves the necessary files.

python train_and_save_model.py

Launch the Web App:
This command starts the interactive application.

streamlit run app.py
