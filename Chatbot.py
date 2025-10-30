import os
from typing import Any, Callable, List, Optional
import importlib

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Allow duplicate OpenMP runtime as a last-resort workaround on macOS builds that otherwise crash.
# The user previously set this. Keep it but prefer avoiding it when possible.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Dynamically import embedding and chat model implementations from either modern or community packages.
OpenAIEmbeddings: Optional[Callable[..., Any]] = None
for mod_name, attr in (("langchain_openai", "OpenAIEmbeddings"), ("langchain_community.embeddings", "OpenAIEmbeddings")):
    try:
        mod = importlib.import_module(mod_name)
        OpenAIEmbeddings = getattr(mod, attr)
        break
    except Exception:
        OpenAIEmbeddings = None

# Vectorstore and QA chain imports (these are stable in langchain)
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

ChatOpenAI: Optional[Callable[..., Any]] = None
for mod_name, attr in (("langchain.chat_models", "ChatOpenAI"), ("langchain_community.chat_models", "ChatOpenAI")):
    try:
        mod = importlib.import_module(mod_name)
        ChatOpenAI = getattr(mod, attr)
        break
    except Exception:
        ChatOpenAI = None

load_dotenv()

class PDFChatbot:
    """A small Streamlit PDF Q&A chatbot wrapper.

    Responsibilities:
    - Read and extract text from an uploaded PDF.
    - Split text into chunks suitable for embeddings.
    - Create an embeddings model and FAISS vectorstore.
    - Run similarity search and answer user questions using a Chat model.

    The class tries to be robust to different langchain packaging versions by dynamically
    locating the OpenAIEmbeddings and ChatOpenAI implementations.
    """

    def __init__(self, openai_api_key: Optional[str] = None) -> None:
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            # export for libraries that read the environment
            os.environ["OPENAI_API_KEY"] = self.openai_api_key

        self.embeddings = None
        self.vector_store = None
        self.text_chunks: List[str] = []

    def _instantiate(self, cls: Callable[..., Any], **kwargs) -> Any:
        """Try to instantiate cls with kwargs, falling back to no-arg construction.

        Many library versions accept an `openai_api_key` kwarg; others rely on env var.
        """
        if not callable(cls):
            raise RuntimeError(f"Provided class {cls!r} is not callable")

        try:
            return cls(**kwargs) if kwargs else cls()
        except TypeError:
            # constructor doesn't accept these kwargs
            return cls()

    def read_pdf(self, file) -> str:
        """Read uploaded Streamlit file-like object and extract text."""
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text

    def split_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return splitter.split_text(text)

    def ensure_embeddings(self) -> None:
        """Create an embeddings instance if possible."""
        if self.embeddings is not None:
            return
        if not OpenAIEmbeddings:
            raise RuntimeError("OpenAIEmbeddings implementation not found. Install 'langchain-openai' or provide 'langchain_community'.")

        try:
            # prefer passing the key if available
            kwargs = {"openai_api_key": self.openai_api_key} if self.openai_api_key else {}
            self.embeddings = self._instantiate(OpenAIEmbeddings, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize embeddings: {exc}")

    def build_vectorstore(self, chunks: List[str]) -> None:
        """Create FAISS vector store from text chunks and embeddings."""
        if not chunks:
            raise ValueError("No chunks provided to build the vector store")
        self.ensure_embeddings()
        try:
            self.vector_store = FAISS.from_texts(chunks, self.embeddings)
        except Exception as exc:
            raise RuntimeError(f"Failed to create FAISS vector store: {exc}")

    def similarity_search(self, query: str, k: int = 4):
        if not self.vector_store:
            raise RuntimeError("Vector store is not initialized")
        return self.vector_store.similarity_search(query, k=k)

    def answer_question(self, matches: List[Any], question: str) -> str:
        """Run a QA chain on the matched documents and return the generated answer string."""
        if not matches:
            return ""
        if not ChatOpenAI:
            raise RuntimeError("ChatOpenAI implementation not found. Install a compatible langchain package.")

        # instantiate LLM (prefer passing api key if available)
        kwargs = {"openai_api_key": self.openai_api_key} if self.openai_api_key else {}
        llm = self._instantiate(ChatOpenAI, temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo", **kwargs)

        chain = load_qa_chain(llm, chain_type="stuff")
        try:
            return chain.run(input_documents=matches, question=question)
        except Exception as exc:
            raise RuntimeError(f"Error running QA chain: {exc}")

    def run_streamlit(self) -> None:
        """Build and run the Streamlit UI. This can be called directly when the module is executed.

        The UI stores heavy objects in st.session_state to avoid recomputing embeddings on each interaction.
        """
        st.header("My First Chatbot")

        with st.sidebar:
            st.title("Your Documents")
            uploaded_file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

        if uploaded_file is None:
            st.info("Upload a PDF in the sidebar to begin")
            return

        # read and process PDF once per upload
        file_id = getattr(uploaded_file, "name", "uploaded_pdf")
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = {}

        if file_id not in st.session_state.processed_files:
            try:
                raw_text = self.read_pdf(uploaded_file)
                if not raw_text or not raw_text.strip():
                    st.error("Could not extract any text from the uploaded PDF. Ensure it contains selectable/textual content.")
                    return

                chunks = self.split_text(raw_text)
                if not chunks:
                    st.error("No text chunks were generated from the document. Adjust chunk_size or check the PDF contents.")
                    return

                # build embeddings and FAISS vector store
                try:
                    self.build_vectorstore(chunks)
                except Exception as exc:
                    st.error(str(exc))
                    return

                st.session_state.processed_files[file_id] = {
                    "chunks": chunks,
                    # store vector_store in session for reuse
                    "vector_store": self.vector_store,
                }
            except Exception as exc:
                st.error(f"Error processing uploaded PDF: {exc}")
                return
        else:
            # reuse stored vector store
            stored = st.session_state.processed_files[file_id]
            self.text_chunks = stored.get("chunks", [])
            self.vector_store = stored.get("vector_store")

        # question input and response
        user_question = st.text_input("Type your question here")
        if not user_question:
            st.info("Type a question to get answers from the uploaded PDF.")
            return

        try:
            matches = self.similarity_search(user_question)
        except Exception as exc:
            st.error(f"Error during similarity search: {exc}")
            return

        if not matches:
            st.warning("No relevant passages found for your question.")
            return

        try:
            answer = self.answer_question(matches, user_question)
        except Exception as exc:
            st.error(str(exc))
            return

        # display only the answer by default
        st.subheader("Answer")
        st.write(answer)

        # provide an expander to show the source passages (raw chunks) if the user wants them
        with st.expander("Show source passages (raw chunks)"):
            for i, doc in enumerate(matches, start=1):
                content = getattr(doc, "page_content", str(doc))
                st.markdown(f"**Passage {i}:**")
                st.write(content)


def main() -> None:
    bot = PDFChatbot()
    bot.run_streamlit()


if __name__ == "__main__":
    main()
