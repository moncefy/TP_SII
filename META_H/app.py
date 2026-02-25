import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("CSV Scatter Plot App")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Data Preview:")
    st.dataframe(df.head())

    col1 = st.selectbox("Select X column", df.columns)
    col2 = st.selectbox("Select Y column", df.columns)

    fig, ax = plt.subplots()
    ax.scatter(df[col1], df[col2])
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)

    st.pyplot(fig)