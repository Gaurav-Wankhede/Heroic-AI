from pandasai.llm.local_llm import LocalLLM
import streamlit as st
from pandasai.connectors import SqliteConnector
from pandasai import SmartDataframe

my_connector = SqliteConnector(
    config={
        "database":"E:\Gaurav's Files And Folders\ACCIOJOBS\PROJECTS\IPL Prediction Analysis\database\IPL_Prediction_Analysis.db",
        "table":"Batting",
    }
)

model = LocalLLM(
    api_base=st.secrets["GROQ_API_KEY"],
    model="llama3"
)

df_connector = SmartDataframe(my_connector, config={"llm": model})

st.title("MySQL with Llama-3")

prompt = st.text_input("Enter your prompt:")

if st.button("Generate"):
    if prompt:
        with st.spinner("Generating response..."):
            st.write(df_connector.chat(prompt))

