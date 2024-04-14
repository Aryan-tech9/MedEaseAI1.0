import json
import os
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqGenerator.utils import read_file, get_table_data
from src.mcqGenerator.logger import logging


from langchain.chat_models import ChatCohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain


load_dotenv()

key = os.getenv("Cohere_api_key")

llm = ChatCohere(cohere_api_key = key, temperature = 0.5)

template = """
    Text: {text}
    You are an expert MCQ maker. Given the above text it is your job to
    create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
    Make sure the questions are not repeated and check all the questions to be conforming the text as well.
    Make sure to format your response like RESPONSE_JSON below and use it as a guide.
    Ensure to make {number} MCQS
    {respnse_json} 
"""

quiz_generation_prompt = PromptTemplate(
    input_variables = ["text", "number", "grade", "tone", "response_json"],
    template = template
)

quiz_chain = LLMChain(llm = llm, prompts = quiz_generation_prompt, output_key = "quiz", verbose = True)

template2 = """
    You are an expert english gramarian and writer. Given a Multiple Choice Quiz for {subject} students.
    You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for the complexity analysis
    if the quiz is not at per with the cognitive and analytical abilities of the students,
    update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
    Quiz_MCQs:
    {quiz}
    Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(input_variables = ["subject", "quiz"], template = template2)

review_chain = LLMChain(llm = llm, prompt = quiz_evaluation_prompt, output_key = "review", verbose = True)

generate_evaluation_chain = SequentialChain(chains = [quiz_chain, review_chain], input_variables = ["text", "number", "subject", "tone", "response_json"],
                                            output_variables = ["quiz", "review"], verbose = True)