#!/usr/bin/env python
# coding: utf-8

# imports
import streamlit as st
import pandas as pd
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle as pkl
import shap


# # Load the saved model
# model = pkl.load(open("model_LR.p","rb"))

# setting
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="loan.png",
    # layout = 'wide',
    initial_sidebar_state = 'expanded'
)

# Disable warning
st.set_option('deprecation.showPyplotGlobalUse', False)


######################
#main page layout
######################

# st.title("Loan Default Prediction💵")
# st.subheader("""Are you sure your loan applicant is surely going to pay the loan back?💸 This machine learning app will help you to make a prediction to help you with your decision!""")


# col1, col2 = st.columns([1, 1])

# with col1:
#     st.image("loan.png")

# with col2:
#     st.write("""To borrow money, credit analysis is performed. Credit analysis involves the measure to investigate
# the probability of the applicant to pay back the loan on time and predict its default/ failure to pay back.

# These challenges get more complicated as the count of applications increases that are reviewed by loan officers.
# Human approval requires extensive hour effort to review each application, however, the company will always seek
# cost optimization and improve human productivity. This sometimes causes human error and bias, as it’s not practical
# to digest a large number of applicants considering all the factors involved.""")


st.subheader("To predict the default probability, you need to follow the steps below:")
st.markdown(
    """
    1. Select the model to use;
    2. Choose the model threhold to determine how conservative the model is;
    3. Enter/choose the parameters that best descibe your applicant on the left side bar;
    4. Press the "Predict" button and wait for the result.
    """
    )

st.subheader("Below you could find prediction result: ")


######################
#sidebar layout
######################

# input the model to use
st.sidebar.title("Select Model")
model_select = st.sidebar.selectbox('Please choose your model', ("Logistic regression", "Random forest"), index = 1)

# Load the saved model
if model_select == "Logistic regression":
    model = pkl.load(open("pages/model_LR.p","rb"))
elif model_select == "Random forest":
    model = pkl.load(open("pages/model_LR.p","rb"))


# input threshold
st.sidebar.title("Model threshold")
st.sidebar.write("The smaller the threshold, the more conservative the model")
threshold = st.sidebar.slider("Threshold:",min_value=0.0, max_value=1.0, step=0.01, value = 0.30)


# input user features
st.sidebar.title("Loan Applicant Info")
# st.sidebar.image("ab.png", width=100)
st.sidebar.write("Please choose parameters that descibe the applicant")

early_return_times = st.sidebar.slider("Early return times:",min_value=0, max_value=5, step=1)
early_return_amount = st.sidebar.slider("Early return amount:",min_value=0, max_value=20000, step=100)
early_return_amount_3mon = st.sidebar.slider("Early return amount in last 3 months:",min_value=0, max_value=10000, step=100)
debt_loan_ratio = st.sidebar.slider("Debt_loan_ratio:",min_value=0, max_value=50, step=1)
interest = st.sidebar.slider("interest:",min_value=0, max_value=50, step=1)
total_loan = st.sidebar.slider("total_loan:",min_value=0, max_value=50000, step=100)
house_exist = st.sidebar.slider("house_exist:",min_value=0, max_value=50, step=1)
known_outstanding_loan = st.sidebar.slider("known_outstanding_loan:",min_value=0, max_value=50, step=1)
monthly_payment = st.sidebar.slider("monthly_payment:",min_value=0, max_value=50, step=1)
credit_class = st.sidebar.selectbox("Credit_class:",("A", "B", "C", "D", "E", "F", 'G'))


######################
#Interpret the result
######################

# preprocess user-input data
def preprocess(early_return_amount, early_return_times, early_return_amount_3mon, debt_loan_ratio, interest, total_loan, house_exist, known_outstanding_loan, monthly_payment, credit_class):
    # Pre-processing user input

    user_input_dict = {
        'early_return':[early_return_times],
        'early_return_amount':[early_return_amount], 
        'early_return_amount_3mon':[early_return_amount_3mon],
        'known_outstanding_loan':[known_outstanding_loan],
        'monthly_payment':[monthly_payment],
        'debt_loan_ratio':[debt_loan_ratio],
        'interest':[interest],
        # 'class' : [1],
        'class':[credit_class],
        'total_loan':[total_loan],
        'house_exist':[house_exist],
        
    }
    user_input = pd.DataFrame(data=user_input_dict)

    cleaner_type = {
        "class": {"A": 1, "B": 2, "C": 3, "D": 4,"E": 5, "F": 6,'G': 7},
    }

    user_input = user_input.replace(cleaner_type)

    return user_input



# user_input = preprocessed data
user_input = preprocess(early_return_amount, early_return_times, early_return_amount_3mon, debt_loan_ratio, interest, total_loan, house_exist, known_outstanding_loan, monthly_payment, credit_class)

# predict button
btn_predict = st.sidebar.button("Predict")

# load the data for shap
loans = st.cache(pd.read_csv)("pages/data_for_shap.csv") # allow_output_mutation=True
# class_dict = {
#     'A': 1,
#     'B': 2,
#     'C': 3,
#     'D': 4,
#     'E': 5,
#     'F': 6,
#     'G': 7,
# }
# loans_data['class'] = loans['class'].map(class_dict)


if btn_predict:
    st.write("Your input:")
    st.write(user_input)
    
    
    pred = model.predict_proba(user_input)[:, 1]

    if pred[0] > threshold:
        st.error('Warning! The applicant has a high risk to not pay the loan back!')
        st.write(f'Probability of default: {round(pred[0],2)}')
    else:
        st.success('It is green! The aplicant has a high probability to pay the loan back!')
        st.write(f'Probability of default: {round(pred[0],2)}')
        
# prepare test set for shap explainability
    # loans = st.cache(pd.read_csv, allow_output_mutation=True)("pages/mycsvfile.csv")
    # class_dict = {
    #     'A': 1,
    #     'B': 2,
    #     'C': 3,
    #     'D': 4,
    #     'E': 5,
    #     'F': 6,
    #     'G': 7,
    # }
    # loans['class'] = loans['class'].map(class_dict)
    X = loans[[
        'early_return','early_return_amount','early_return_amount_3mon',
        'known_outstanding_loan','monthly_payment','debt_loan_ratio','interest',
        'class','total_loan','house_exist'
        ]]
    y = loans[['isDefault']]
    y_ravel = y.values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y_ravel, test_size=0.2, random_state=2023, stratify=y)
    
    st.subheader('Result Interpretability - Applicant Level')
    shap.initjs()
    explainer = shap.Explainer(model, X) # explainer = shap.Explainer(model, X_train)
    shap_values = explainer(user_input)
    fig = shap.plots.bar(shap_values[0])
    st.pyplot(fig)

    st.subheader('Model Interpretability - Overall')
    shap_values_ttl = explainer(X) #  shap_values_ttl = explainer(X_test)
    fig_ttl = shap.plots.beeswarm(shap_values_ttl)
    st.pyplot(fig_ttl)
 


