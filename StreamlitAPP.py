import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqGenerator.utils import read_file, get_table_data
import streamlit as st
from langchain.callbacks import StdOutCallbackHandler
from src.mcqGenerator.MCQGenerator import generate_evaluation_chain

from src.mcqGenerator.logger import logging


#loading json file
with open('C:\Users\sarga\MCQGenerator\Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)
    
# stramlit for creating the web application
st.title("MCQ Creator Application using Langchain")

#creating a form using streamlit.form

with st.form("user_inputs"):
    #file upload
    uploaded_file = st.file_uploader("Upload a pdf or text file")
    
    #input fields
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50)
    
    #subject
    subject =st.text_input("Insert Subject", max_chars=20)
    
    #Quiz tone
    tone = st.text_input("Complexity Level of Questions", max_chars=20, placeholder="Simple")
    
    #Add button
    button = st.form_submit_button("Create MCQs")
    
    #check if the button is created all the fields are have input
    
    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading..."):
            try:
                text = read_file(uploaded_file)
                #find token and cost of API call
                # Create an instance of StdOutCallbackHandler
                stdout_callback = StdOutCallbackHandler()

                # Generate evaluation chain using the callback instance
                response=generate_evaluation_chain(
                {
                "text": text,
                "number": mcq_count,
                "subject":subject,
                "tone": tone,
                "response_json": json.dumps(RESPONSE_JSON)
                }
                )
            
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error")
                
            else:
                if isinstance(response, dict):
                    quiz = response.get("quiz", None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index+1
                            st.table(df)
                            
                            #Display the review in a text as well
                            st.text_area(label="Review", value = response["review"])
                        else:
                            st.error("Error in the table data")
                else:
                    st.write(response)