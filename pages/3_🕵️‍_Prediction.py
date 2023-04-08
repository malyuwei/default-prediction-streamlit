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
    1. Choose the model threhold to determine how conservative the model is;
    2. Choose the parameters that best descibe your applicant on the left side bar;
    3. Press the "Predict" button and wait for the result.
    """
    )



######################
#sidebar layout
######################

# input the model to use
# st.sidebar.title("Select Model")
# model_select = st.sidebar.selectbox('Please choose your model', ("Logistic regression", "Random forest"), index = 1)

# Load the saved model
model = pkl.load(open("pages/model_RF.p","rb"))
# if model_select == "Logistic regression":
#     model = pkl.load(open("pages/model_LR.p","rb"))
# elif model_select == "Random forest":
#     model = pkl.load(open("pages/model_LR.p","rb"))


# input threshold
st.sidebar.title("Model threshold")
st.sidebar.write("The smaller the threshold, the more conservative the model")
threshold = st.sidebar.slider("Threshold:",min_value=0.0, max_value=1.0, step=0.01, value = 0.30)


# input user features
st.sidebar.title("Loan Applicant Info")
# st.sidebar.image("ab.png", width=100)
st.sidebar.write("Please choose parameters that descibe the applicant")

early_return_times = st.sidebar.slider("Early return times:",min_value=0, max_value=5, step=1, value = 1)
# early_return_amount = st.sidebar.slider("Early return amount:",min_value=0, max_value=20000, step=100)
early_return_amount_3mon = st.sidebar.slider("Early return amount in last 3 months:",min_value=0, max_value=10000, step=100, value = 2000)
debt_loan_ratio = st.sidebar.slider("Debt loan ratio:",min_value=0, max_value=50, step=1, value=8)
interest = st.sidebar.slider("Interest:",min_value=3, max_value=50, step=1, value = 5)
total_loan = st.sidebar.slider("Total loan:",min_value=0, max_value=60000, step=100, value=30000)
house_exist = st.sidebar.slider("House exist:",min_value=0, max_value=5, step=1, value = 1)
known_outstanding_loan = st.sidebar.slider("Known outstanding loan:",min_value=0, max_value=50, step=1)
monthly_payment = st.sidebar.slider("Monthly payment:",min_value=0, max_value=3000, step=100, value = 1000)
credit_class = st.sidebar.selectbox("Credit class:",("A", "B", "C", "D", "E", "F", 'G'), index = 1)



# term = st.sidebar.radio("Select Loan term: ", ('36months', '60months'))
# loan_amnt =st.sidebar.slider("Please choose Loan amount you would like to apply:",min_value=1000, max_value=40000,step=500)
# emp_length = st.sidebar.selectbox('Please choose your employment length', ("< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
#                                                                            "6 years", "7 years","8 years","9 years","10+ years") )
# annual_inc =st.sidebar.slider("Please choose your annual income:", min_value=10000, max_value=200000,step=1000)
# sub_grade =st.sidebar.selectbox('Please choose grade', ("A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3","B4","B5","C1", "C2", "C3",
#                                                         "C4", "C5", "D1", "D2", "D3","D4","D5", "E1", "E2", "E3","E4","E5","F1", "F2",
#                                                         "F3", "F4", "F5", "G1", "G2", "G3","G4","G5"))
# dti=st.sidebar.slider("Please choose DTI:",min_value=0.1, max_value=100.1,step=0.1)
# mths_since_recent_inq=st.sidebar.slider("Please choose your mths_since_recent_inq:",min_value=1, max_value=25,step=1)
# revol_util=st.sidebar.slider("Please choose revol_util:",min_value=0.1, max_value=150.1,step=0.1)
# num_op_rev_tl=st.sidebar.slider("Please choose num_op_rev_tl:",min_value=1, max_value=50,step=1)


######################
#Interpret the result
######################

# preprocess user-input data
def preprocess(early_return_times, early_return_amount_3mon, debt_loan_ratio, interest, total_loan, house_exist, known_outstanding_loan, monthly_payment, credit_class):
    # Pre-processing user input

    user_input_dict = {
        'early_return':[early_return_times],
        # 'early_return_amount':[early_return_amount], 
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
user_input = preprocess(early_return_times, early_return_amount_3mon, debt_loan_ratio, interest, total_loan, house_exist, known_outstanding_loan, monthly_payment, credit_class)

cf = st.sidebar.selectbox("Choose a feature for denpendence plot", (user_input.columns),5)

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
X = loans[[
        'early_return',
        # 'early_return_amount',
        'early_return_amount_3mon',
        'known_outstanding_loan','monthly_payment','debt_loan_ratio','interest',
        'class','total_loan','house_exist'
        ]]
y = loans[['isDefault']]
y_ravel = y.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y_ravel, test_size=0.2, random_state=2023, stratify=y)

if btn_predict:

    # placeholder1 = st.empty()
    # with placeholder1.container():
    #     f1,f2 = st.columns(2)
    #     with f1:
    #         st.write("#### Your input")
    #         st.write(user_input)

    #         pred = model.predict_proba(user_input)[:, 1]

    #         if pred[0] > threshold:
    #             st.error('Warning! The applicant has a high risk to not pay the loan back!')
    #             st.write(f'Probability of default: {round(pred[0],2)}')
    #         else:
    #             st.success('It is green! The aplicant has a high probability to pay the loan back!')
    #             st.write(f'Probability of default: {round(pred[0],2)}')
    #     with f2:
    #         from shap.plots import _waterfall
    #         shap.initjs()
    #         explainer = shap.TreeExplainer(model,X)
    #         shap_values = explainer(user_input)
    #         # st.write(shap_values)
    #         # st.write(explainer.expected_value[1])
    #         # st.write(shap_values.values[0][:,1])
    #         # st.write(shap_values[0])
    #         waterfall = _waterfall.waterfall_legacy(explainer.expected_value[1], shap_values.values[0][:,1], X_test.iloc[0])
    #         # waterfall = shap.plots.waterfall(explainer.expected_value[1],shap_values.values)
    #         # shap.plots.waterfall(0.15,shap_values[0])
    #         # shap.plots.waterfall(shap_values[0][0])
    #         st.pyplot(waterfall)
    #         # st.pyplot(bbox_inches='tight')
    st.subheader("Your input")
    st.write(user_input)


    st.subheader("The prediction result: ")
    pred = model.predict_proba(user_input)[:, 1]
    if pred[0] > threshold:
        st.error('The applicant has a high risk to not pay the loan back!')
        st.write(f'Probability of default: {round(pred[0],2)}')
    else:
        st.success('The aplicant has a high probability to pay the loan back!')
        st.write(f'Probability of default: {round(pred[0],2)}')


    st.subheader('Model Interpretability - Applicant Level')
    from shap.plots import _waterfall
    shap.initjs()
    explainer = shap.TreeExplainer(model,X)
    shap_values = explainer(user_input)
    waterfall = _waterfall.waterfall_legacy(explainer.expected_value[1], shap_values.values[0][:,1], X_test.iloc[0])
    st.pyplot(waterfall)
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
    # X = loans[[
    #     'early_return','early_return_amount','early_return_amount_3mon',
    #     'known_outstanding_loan','monthly_payment','debt_loan_ratio','interest',
    #     'class','total_loan','house_exist'
    #     ]]
    # y = loans[['isDefault']]
    # y_ravel = y.values.ravel()

    # X_train, X_test, y_train, y_test = train_test_split(X, y_ravel, test_size=0.2, random_state=2023, stratify=y)
    

    st.subheader('Model Interpretability - Overall Level')
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X, check_additivity=False)
    # shap绝对值平均值-特征重要性
    fig_importance = shap.summary_plot(shap_values[1], X, plot_type = 'bar', plot_size = (10,8))
    st.pyplot(fig_importance)

    # summary_plot
    fig_cellular = shap.summary_plot(shap_values[1], X, plot_size = (10,6)) # plot_type = 'violin'
    st.pyplot(fig_cellular)

    # Dependence plot for features
    fig_denpendence = shap.dependence_plot(cf, shap_values[1], X, interaction_index=None)
    st.pyplot(fig_denpendence)

