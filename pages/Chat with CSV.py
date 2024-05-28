import pandas as pd
import streamlit as st
from langchain_groq import ChatGroq
from pandasai import SmartDataframe

# Set Streamlit page configuration
st.set_page_config(page_icon="💬", layout="wide", page_title="🐱‍💻 Heroic AI...")

# Initialize the LLM
llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model="LLaMA3-70b-8192",  # Pass the model name as a string
)

# Set title
st.title("Multiple-CSV Heroic ChatApp powered by LLM")

# File uploader for multiple CSV files
uploaded_files = st.file_uploader("Choose CSV files", type=["csv"], accept_multiple_files=True)

# Initialize an empty DataFrame to store all data
all_data = pd.DataFrame()

if uploaded_files:
    # Load and concatenate all CSV files
    data_frames = []
    for uploaded_file in uploaded_files:
        try:
            data = pd.read_csv(uploaded_file)
            data_frames.append(data)
            st.write(f"Loaded {uploaded_file.name} with {data.shape[0]} rows and {data.shape[1]} columns.")
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")

    if data_frames:
        all_data = pd.concat(data_frames, ignore_index=True)
        st.write("Combined Data Preview:")
        st.write(all_data.head(10))
    else:
        st.warning("No valid CSV files were uploaded.")
else:
    st.info("Please upload one or more CSV files.")

# Create a SmartDataframe instance if data is available
if not all_data.empty:
    sdf = SmartDataframe(config={"llm": llm}, df=all_data)  # Update with concatenated DataFrame

    # Chat input for user queries
    query = st.text_input("Ask a question about the data...")

    if query:
        try:
            result = sdf.chat(query)
            st.write(result)
        except Exception as e:
            st.error(f"Error processing the query: {e}")
else:
    st.info("No data available to query.")
