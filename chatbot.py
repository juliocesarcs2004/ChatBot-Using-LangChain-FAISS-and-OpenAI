import os

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import importlib
from typing import Any, Callable, Optional, cast

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Dynamically try to load OpenAIEmbeddings from modern or legacy packages.
OpenAIEmbeddings: Optional[Callable[..., Any]] = None
for mod_name, attr in (("langchain_openai", "OpenAIEmbeddings"), ("langchain_community.embeddings", "OpenAIEmbeddings")):
    try:
        mod = importlib.import_module(mod_name)
        OpenAIEmbeddings = getattr(mod, attr)
        break
    except Exception:
        OpenAIEmbeddings = None

# FAISS and QA chain imports
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# Dynamically load ChatOpenAI from the modern langchain package or the community package
ChatOpenAI: Optional[Callable[..., Any]] = None
for mod_name, attr in (("langchain.chat_models", "ChatOpenAI"), ("langchain_community.chat_models", "ChatOpenAI")):
    try:
        mod = importlib.import_module(mod_name)
        ChatOpenAI = getattr(mod, attr)
        break
    except Exception:
        ChatOpenAI = None

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ensure downstream libraries that expect env var can read it
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

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
        # guard against pages with no extractable text
        page_text = page.extract_text()
        if page_text:
            text += page_text
        #st.write(text)

    # If we couldn't extract any text, show an error and skip processing
    if not text or not text.strip():
        st.error("Could not extract any text from the uploaded PDF. Try a different file or ensure the PDF contains selectable text (not just scanned images).")
    else:
        #Break it into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n"],
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        #st.write(chunks)

        if not chunks:
            st.error("No text chunks were generated from the document. Adjust chunk_size or check the PDF contents.")
        else:
            # Ensure embeddings class is available
            if not callable(OpenAIEmbeddings):
                st.error("OpenAIEmbeddings couldn't be imported. Install 'langchain-openai' or ensure 'langchain_community' is available.")
            else:
                #generating embeddings
                try:
                    # Ensure runtime safety and help static analyzers by casting to a Callable
                    if not callable(OpenAIEmbeddings):
                        raise RuntimeError("Embeddings implementation is not callable")
                    EmbeddingsCls = cast(Callable[..., Any], OpenAIEmbeddings)
                    embeddings = EmbeddingsCls()
                except Exception as e:
                    st.error(f"Error initializing embeddings: {e}")
                    embeddings = None

                if embeddings is not None:
                    #creating vector store - FAISS
                    try:
                        vector_store = FAISS.from_texts(chunks, embeddings)
                    except Exception as e:
                        st.error(f"Error creating FAISS vector store: {e}")
                        vector_store = None

                    #get user question
                    user_question = st.text_input("Type your question here")

                    # Ensure match is defined so we don't get NameError later
                    match = None

                    #do similarity search
                    if user_question and vector_store is not None:
                        try:
                            match = vector_store.similarity_search(user_question)
                            # Do not display chunks here. If there are no matches, inform the user.
                            if not match:
                                st.warning("No relevant passages found for your question.")
                        except Exception as e:
                            st.error(f"Error during similarity search: {e}")
                            match = None

                    # Only initialize the LLM and run the QA chain if we have both a question and search results
                    if user_question and match:
                        # Check ChatOpenAI is available and callable
                        if not callable(ChatOpenAI):
                            st.error("ChatOpenAI couldn't be imported. Install a compatible 'langchain' package or 'langchain_community'.")
                        else:
                            # Ensure runtime safety and help static analyzers by casting to a Callable
                            if not callable(ChatOpenAI):
                                raise RuntimeError("ChatOpenAI implementation is not callable")
                            ChatOpenAICls = cast(Callable[..., Any], ChatOpenAI)
                            #define the LLM
                            llm = ChatOpenAICls(temperature=0)

                            #output results (show only the generated answer by default)
                            chain = load_qa_chain(llm, chain_type="stuff")
                            try:
                                response = chain.run(input_documents=match, question=user_question)
                                st.write(response)

                                # After showing the answer, provide an optional expander for sources.
                                with st.expander("Show source passages (raw chunks)"):
                                    if match:
                                        for i, doc in enumerate(match, start=1):
                                            content = getattr(doc, "page_content", str(doc))
                                            st.write(f"Passage {i}:")
                            except Exception as e:
                                st.error(f"Error running QA chain: {e}")
                    else:
                        # Inform the user to type a question if none was provided, or show a message if no matches
                        if not user_question:
                            st.info("Type a question to get answers from the uploaded PDF.")
                        elif not match:
                            st.warning("No relevant passages found for your question.")
