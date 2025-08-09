import streamlit as st
import pandas as pd
import joblib

# Load the trained model, scaler, and model columns
try:
    model = joblib.load('churn_model.joblib')
    scaler = joblib.load('scaler.joblib')
    model_columns = joblib.load('model_columns.joblib')
except FileNotFoundError:
    st.error("Model files not found. Please run the `train_and_save_model.py` script first.")
    st.stop()

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Customer Churn Prediction", page_icon="👋", layout="centered")
st.title("Customer Churn Prediction 🔮")
st.write("This app predicts the likelihood of a customer churning based on their account information and usage patterns.")

# --- User Input Sidebar ---
st.sidebar.header("Customer Information")

def user_input_features():
    international_plan = st.sidebar.selectbox('International Plan', ('no', 'yes'))
    voice_mail_plan = st.sidebar.selectbox('Voice Mail Plan', ('no', 'yes'))
    number_vmail_messages = st.sidebar.slider('Number of Voicemail Messages', 0, 51, 0)
    total_day_minutes = st.sidebar.slider('Total Day Minutes', 0.0, 351.0, 180.0)
    total_day_calls = st.sidebar.slider('Total Day Calls', 0, 165, 100)
    total_eve_minutes = st.sidebar.slider('Total Evening Minutes', 0.0, 364.0, 200.0)
    total_eve_calls = st.sidebar.slider('Total Evening Calls', 0, 170, 100)
    total_night_minutes = st.sidebar.slider('Total Night Minutes', 0.0, 395.0, 200.0)
    total_night_calls = st.sidebar.slider('Total Night Calls', 0, 175, 100)
    total_intl_minutes = st.sidebar.slider('Total International Minutes', 0.0, 20.0, 10.0)
    total_intl_calls = st.sidebar.slider('Total International Calls', 0, 20, 4)
    number_customer_service_calls = st.sidebar.slider('Customer Service Calls', 0, 9, 1)

    data = {
        'international_plan': 1 if international_plan == 'yes' else 0,
        'voice_mail_plan': 1 if voice_mail_plan == 'yes' else 0,
        'number_vmail_messages': number_vmail_messages,
        'total_day_minutes': total_day_minutes,
        'total_day_calls': total_day_calls,
        'total_eve_minutes': total_eve_minutes,
        'total_eve_calls': total_eve_calls,
        'total_night_minutes': total_night_minutes,
        'total_night_calls': total_night_calls,
        'total_intl_minutes': total_intl_minutes,
        'total_intl_calls': total_intl_calls,
        'number_customer_service_calls': number_customer_service_calls,
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Prediction Logic ---
if st.sidebar.button('Predict Churn Likelihood'):
    # Ensure final_df matches training feature structure
    final_df = pd.get_dummies(input_df).reindex(columns=model_columns, fill_value=0)

    # Ensure scaler columns match exactly what was seen at fit time
    expected_scaler_cols = scaler.feature_names_in_
    for col in expected_scaler_cols:
        if col not in final_df:
            final_df[col] = 0
    final_df = final_df[expected_scaler_cols]

    # Scale numerical features
    final_df[expected_scaler_cols] = scaler.transform(final_df[expected_scaler_cols])

    # Get prediction probability
    churn_probability = model.predict_proba(final_df)[:, 1]
    churn_prob_percentage = churn_probability[0] * 100

    st.subheader("Prediction Result")
    st.progress(int(churn_prob_percentage))
    st.metric(label="Churn Likelihood", value=f"{churn_prob_percentage:.2f}%")

    if churn_prob_percentage >= 50:
        st.error("High Risk: This customer is likely to churn. 🚨")
        st.info("Consider reaching out with a retention offer or support.")
    else:
        st.success("Low Risk: This customer is likely to stay. ✅")
