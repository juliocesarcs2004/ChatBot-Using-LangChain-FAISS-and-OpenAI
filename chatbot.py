import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

OPEN_API_KEY = "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"

#Upload PDF Files
st.header("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text+=page.extract_text()
        #st.write(text)

#Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)


#generating embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPEN_API_KEY)

#creating vector store - FAISS
vector_store = FAISS.from_texts(chunks, embeddings)









