import streamlit as st

st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="loan.png",
)

# Disable warning
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Loan Default Prediction Web App💵")
st.write("### Welcome to our app! 👋")
st.write("#### Are you confident that your loan applicant will be able to repay the loan on time? \n ")
st.write("#### This machine learning application can assist you in making a prediction to aid in your decision-making process!")
st.image("personal loan.jpg")
st.write(
"""
Credit analysis is a critical step in borrowing money, as it involves evaluating an applicant's ability to repay the loan on time and predicting the likelihood of default. However, as the number of loan applications increases, it becomes increasingly challenging for loan officers to thoroughly review each application. To optimize costs and improve productivity, companies often look for ways to streamline the process. However, this may lead to errors and bias, as it is difficult for humans to review a large number of applications while considering all relevant factors.

Credit analysis involves assessing an applicant's creditworthiness by examining various factors, such as their credit history, income, debt-to-income ratio, and other financial information. The aim of credit analysis is to make informed decisions about lending money and to minimize the risk of loan defaults. By evaluating these factors, lenders can determine the level of risk involved in lending to the applicant and decide whether to approve or deny the loan request.
""")

# col1, col2 = st.columns([1, 1])

# with col1:
#     st.image("personal loan.jpg")

# with col2:
#     st.write("""To borrow money, credit analysis is performed. Credit analysis involves the measure to investigate
# the probability of the applicant to pay back the loan on time and predict its default/ failure to pay back.

# These challenges get more complicated as the count of applications increases that are reviewed by loan officers.
# Human approval requires extensive hour effort to review each application, however, the company will always seek
# cost optimization and improve human productivity. This sometimes causes human error and bias, as it’s not practical
# to digest a large number of applicants considering all the factors involved.""")