import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff

st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="loan.png",
    layout = 'wide',
)

st.title("Explore the data and find patterns!")

# Disable warning
st.set_option('deprecation.showPyplotGlobalUse', False)


# Explore the dataset
# df = st.cache(pd.read_csv)("pages/train_public.csv")

df = pd.read_csv("pages/train_public.csv",encoding="utf-8")
df["debt_loan_ratio"] = df["debt_loan_ratio"].clip(0, 77)



st.subheader('Dataset Preview')
# if st.button('View 5 random applicant data'):
st.write('View 5 random applicant data')
st.write(df.sample(5))

st.subheader('Data Distribution')
unbalancedf = pd.DataFrame(df.isDefault.value_counts())
st.write('0️ means Not Default, 1️ means Defualt')
st.write(unbalancedf)


f1,f2,f3 = st.columns(3)
with f1:
	st.image("eda plot/plot1.png")
	# a10 = df[df['isDefault'] == 0]['interest']
	# a11 = df[df['isDefault'] == 1]['interest']
	# hist_data = [a10, a11]
	# fig = ff.create_distplot(hist_data,group_labels = ['Not Default', 'Default'])
	# fig.update_layout(title_text='Interest')
	# st.plotly_chart(fig, use_container_width=True)

with f2:
	st.image("eda plot/plot2.png")
	# a20 = df[df['isDefault'] == 0]['debt_loan_ratio']
	# a21 = df[df['isDefault'] == 1]['debt_loan_ratio']
	# hist_data = [a20, a21]
	# fig = ff.create_distplot(hist_data,group_labels = ['Not Default', 'Default'])
	# fig.update_layout(title_text='debt_loan_ratio')
	# st.plotly_chart(fig, use_container_width=True)

with f3:
	st.image("eda plot/plot3.png")
	# a30 = df[df['isDefault'] == 0]['known_outstanding_loan']
	# a31 = df[df['isDefault'] == 1]['known_outstanding_loan']
	# hist_data = [a30, a31]
	# fig = ff.create_distplot(hist_data,group_labels = ['Not Default', 'Default'])
	# fig.update_layout(title_text='known_outstanding_loan')
	# st.plotly_chart(fig, use_container_width=True)

f1,f2,f3 = st.columns(3)
with f1:
	st.image("eda plot/plot4.png")
	# a10 = df[df['isDefault'] == 0]['early_return']
	# a11 = df[df['isDefault'] == 1]['early_return']
	# hist_data = [a10, a11]
	# fig = ff.create_distplot(hist_data,group_labels = ['Not Default', 'Default'])
	# fig.update_layout(title_text='early_return')
	# st.plotly_chart(fig, use_container_width=True)

with f2:
	# st.image("eda plot/plot5.png")
	a20 = df[df['isDefault'] == 0]['early_return_amount_3mon']
	a21 = df[df['isDefault'] == 1]['early_return_amount_3mon']
	hist_data = [a20, a21]
	fig = ff.create_distplot(hist_data,group_labels = ['Not Default', 'Default'])
	fig.update_layout(title_text='early_return_amount_3mon')
	st.plotly_chart(fig, use_container_width=True)

with f3:
	# st.image("eda plot/plot6.png")
	a30 = df[df['isDefault'] == 0]['total_loan']
	a31 = df[df['isDefault'] == 1]['total_loan']
	hist_data = [a30, a31]
	fig = ff.create_distplot(hist_data,group_labels = ['Not Default', 'Default'])
	fig.update_layout(title_text='total_loan')
	st.plotly_chart(fig, use_container_width=True)

st.image("eda plot/plot7.png")
st.image("eda plot/plot8.png")

# f1,f2 = st.columns(2)
# with f1:
# 	st.image("eda plot/plot7.png",use_column_width=True)
# with f2:
# 	st.image("eda plot/plot8.png",use_column_width=True)


st.subheader('Correlation')
st.image("eda plot/plot9.png")