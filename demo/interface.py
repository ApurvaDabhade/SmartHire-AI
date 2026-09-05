import sys, os
sys.dont_write_bytecode = True

import time
from dotenv import load_dotenv

import pandas as pd
import streamlit as st
from openai import OpenAI
from streamlit_modal import Modal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

from llm_agent import ChatBot
from ingest_data import ingest, read_resume_csv, format_import_report
from retriever import SelfQueryRetriever
import chatbot_verbosity as chatbot_verbosity

load_dotenv(override=True)

DATA_PATH = os.getenv("DATA_PATH")
FAISS_PATH = os.getenv("FAISS_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
DEPRECATED_GROQ_MODELS = {
  "llama-3.3-70b-versatile",
  "llama-3.1-8b-instant",
}
GROQ_MODEL = os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b"
if GROQ_MODEL in DEPRECATED_GROQ_MODELS:
  GROQ_MODEL = "openai/gpt-oss-20b"

print(DATA_PATH)
print(FAISS_PATH)

welcome_message = """
  #### Introduction 🚀

  The system is a RAG pipeline designed to assist hiring managers in searching for the most suitable candidates out of thousands of resumes more effectively. ⚡

  The idea is to use a similarity retriever to identify the most suitable applicants with job descriptions.
  This data is then augmented into an LLM generator for downstream tasks such as analysis, summarization, and decision-making. 

  #### Getting started 🛠️

  1. To set up, please add your OpenAI's API key. 🔑 
  2. Type in a job description query. 💬

  Hint: The knowledge base of the LLM has been loaded with a pre-existing vectorstore of [resumes](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/blob/main/data/main-data/synthetic-resumes.csv) to be used right away. 
  In addition, you may also find example job descriptions to test [here](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/blob/main/data/supplementary-data/job_title_des.csv).

  Please make sure to check the sidebar for more useful information. 💡
"""

info_message = """
  # Information

  ### 1. What if I want to use my own resumes?

  If you want to load in your own resumes file, simply use the uploading button above. 
  Upload a CSV of resumes. Preferred columns are `Resume` and `ID`; other common column names are mapped automatically. 

  Keep in mind that the indexing process can take **quite some time** to complete. ⌛

  ### 2. What if I want to set my own parameters?

  You can change the RAG mode and the GPT's model type using the sidebar options above. 

  About the other parameters such as the generator's *temperature* or retriever's *top-K*, I don't want to allow modifying them for the time being to avoid certain problems. 
  FYI, the temperature is currently set at `0.1` and the top-K is set at `5`.  

  ### 3. Is my uploaded data safe? 

  Your data is not being stored anyhow by the program. Everything is recorded in a Streamlit session state and will be removed once you refresh the app. 

  However, it must be mentioned that the **uploaded data will be processed directly by OpenAI's GPT**, which I do not have control over. 
  As such, it is highly recommended to use the default synthetic resumes provided by the program. 

  ### 4. How does the chatbot work? 

  The Chatbot works a bit differently to the original structure proposed in the paper so that it is more usable in practical use cases.

  For example, the system classifies the intent of every single user prompt to know whether it is appropriate to toggle RAG retrieval on/off. 
  The system also records the chat history and chooses to use it in certain cases, allowing users to ask follow-up questions or tasks on the retrieved resumes.
"""

about_message = """
  # About

  This small program is a prototype designed out of pure interest as additional work for the author's Bachelor's thesis project. 
  The aim of the project is to propose and prove the effectiveness of RAG-based models in resume screening, thus inspiring more research into this field.

  The program is very much a work in progress. I really appreciate any contribution or feedback on [GitHub](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline).

  If you are interested, please don't hesitate to give me a star. ⭐
"""


st.set_page_config(page_title="Resume Screening GPT")
st.title("Resume Screening GPT")

if "chat_history" not in st.session_state:
  st.session_state.chat_history = [AIMessage(content=welcome_message)]

if "df" not in st.session_state:
  st.session_state.df = pd.read_csv(DATA_PATH)

if "embedding_model" not in st.session_state:
  st.session_state.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})

if "rag_pipeline" not in st.session_state:
  vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
  st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)

if "resume_list" not in st.session_state:
  st.session_state.resume_list = []

if "api_key" not in st.session_state:
  st.session_state.api_key = GROQ_API_KEY

if "gpt_selection" not in st.session_state or st.session_state.gpt_selection in DEPRECATED_GROQ_MODELS or st.session_state.gpt_selection == GROQ_MODEL:
  st.session_state.gpt_selection = "gpt-4o-mini"

MODEL_ALIASES = {
  "gpt-4o-mini": GROQ_MODEL,
  "gpt-4o": "openai/gpt-oss-120b",
  "gpt-3.5-turbo": GROQ_MODEL,
  "gpt-35-turbo": GROQ_MODEL,
}


def resolve_model(model_name: str) -> str:
  return MODEL_ALIASES.get(model_name, model_name)



def load_default_resumes():
  st.session_state.df = pd.read_csv(DATA_PATH)
  vectordb = FAISS.load_local(
    FAISS_PATH,
    st.session_state.embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
    allow_dangerous_deserialization=True,
  )
  st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)


def set_resume_index(df, vectordb):
  st.session_state.df = df
  st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, df)


def upload_file():
  modal = Modal(key="Demo Key", title="File Error", max_width=500)
  uploaded_file = st.session_state.uploaded_file
  if uploaded_file is None:
    load_default_resumes()
    return

  try:
    if not str(getattr(uploaded_file, "name", "")).lower().endswith(".csv"):
      raise ValueError("Please upload a CSV file only.")
    df_load = read_resume_csv(uploaded_file)
    report = getattr(df_load, "attrs", {}).get("import_report") or {}
    with st.toast("Indexing the uploaded CSV. This may take a while..."):
      vectordb = ingest(df_load, "Resume", st.session_state.embedding_model)
      set_resume_index(df_load, vectordb)
    if report:
      st.session_state.csv_import_message = format_import_report(report)
  except Exception as error:
    with modal.container():
      st.markdown("The uploaded file could not be processed. Please check your CSV file again.")
      st.error(error)
    load_default_resumes()


def groq_client(api_key: str):
  return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def check_openai_api_key(api_key: str):
  try:
    groq_client(api_key).models.list()
    return True
  except Exception:
    return False
  
  
def check_model_name(model_name: str, api_key: str):
  try:
    model_list = [model.id for model in groq_client(api_key).models.list()]
    return resolve_model(model_name) in model_list or model_name in model_list
  except Exception:
    return False


def clear_message():
  st.session_state.resume_list = []
  st.session_state.chat_history = [AIMessage(content=welcome_message)]



user_query = st.chat_input("Type your message here...")

with st.sidebar:
  st.markdown("# Control Panel")

  st.text_input("OpenAI's API Key", type="password", key="api_key")
  st.selectbox("RAG Mode", ["Generic RAG", "RAG Fusion"], placeholder="Generic RAG", key="rag_selection")
  st.text_input("GPT Model", "gpt-4o-mini", key="gpt_selection")
  st.file_uploader("Upload resumes", type=["csv"], key="uploaded_file", on_change=upload_file)
  if st.session_state.get("csv_import_message"):
    st.caption(st.session_state.csv_import_message)
  st.button("Clear conversation", on_click=clear_message)

  st.divider()
  st.markdown(info_message)

  st.divider()
  st.markdown(about_message)
  st.markdown("Made by [Hungreeee](https://github.com/Hungreeee)")


for message in st.session_state.chat_history:
  if isinstance(message, AIMessage):
    with st.chat_message("AI"):
      st.write(message.content)
  elif isinstance(message, HumanMessage):
    with st.chat_message("Human"):
      st.write(message.content)
  else:
    with st.chat_message("AI"):
      message[0].render(*message[1:])


if not st.session_state.api_key:
  st.info("Please add your OpenAI API key to continue. Learn more about [API keys](https://platform.openai.com/api-keys).")
  st.stop()

if not check_openai_api_key(st.session_state.api_key):
  st.error("The API key is incorrect. Please set a valid OpenAI API key to continue. Learn more about [API keys](https://platform.openai.com/api-keys).")
  st.stop()

if not check_model_name(st.session_state.gpt_selection, st.session_state.api_key):
  st.error("The model you specified does not exist. Learn more about [OpenAI models](https://platform.openai.com/docs/models).")
  st.stop()


retriever = st.session_state.rag_pipeline

llm = ChatBot(
  api_key=st.session_state.api_key,
  model=resolve_model(st.session_state.gpt_selection),
)

if user_query is not None and user_query != "":
  with st.chat_message("Human"):
    st.markdown(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))

  with st.chat_message("AI"):
    start = time.time()
    with st.spinner("Generating answers..."):
      document_list = retriever.retrieve_docs(user_query, llm, st.session_state.rag_selection)
      if not isinstance(document_list, list) or not document_list:
        if retriever.meta_data["query_type"] == "no_retrieve" and isinstance(st.session_state.resume_list, list):
          document_list = st.session_state.resume_list
        else:
          document_list = []
      else:
        st.session_state.resume_list = document_list
      query_type = retriever.meta_data["query_type"]
      chat_history = [
        message.content[:400] for message in st.session_state.chat_history
        if isinstance(message, (AIMessage, HumanMessage)) and message.content != welcome_message
      ][-4:]
      stream_message = llm.generate_message_stream(user_query, document_list, chat_history, query_type)
    end = time.time()

    try:
      response = st.write_stream(stream_message)
    except Exception as error:
      error_text = str(error)
      if "413" in error_text or "rate_limit_exceeded" in error_text or "Request too large" in error_text:
        response = "The request was too large for the current API limit. Please try a shorter job description, or wait a minute and send the query again."
        st.error(response)
      else:
        raise
    
    retriever_message = chatbot_verbosity
    retriever_message.render(document_list, retriever.meta_data, end-start)

    st.session_state.chat_history.append(AIMessage(content=response))
    st.session_state.chat_history.append((retriever_message, document_list, retriever.meta_data, end-start))