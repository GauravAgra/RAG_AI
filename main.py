import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

st.title("PDF RAG Agent with LangChain and Ollama")

# Load doc
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])


@st.cache_resource
def create_vector_db(pdf_file):
    # Create a python PDF objectt
    pdf_reader = PdfReader(pdf_file)

    # Variable to store PDF text
    text = ""

    # Iterates pages and store content in text
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"

    # Initialize splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # Split content in chunks
    chunks = splitter.split_text(text)

    # Convert chunks into doc
    doc = [Document(page_content=chunk) for chunk in chunks]

    # Initialize embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Store chunks in a vector database
    vector_db = Chroma.from_documents(doc, embeddings)

    return vector_db


# Upload doc
if pdf_file is not None:
    # Call vector db
    vector_db = create_vector_db(pdf_file)

    # Initialize retriever
    retriever = vector_db.as_retriever()

    # Initialize llm
    llm = ChatOllama(model="gemma3:1b", temperature=0)

    # Ask Question
    question = st.text_input("Enter a question")

    # Do similarity search
    retrieved_doc = retriever.invoke(question)

    prompt = f"""
    Answer the below question only using the below provided context.
    
    Context
    {retrieved_doc}
    
    Question
    {question} 
    """
    # Invoke llm and Pass question and result of similarity search
    response = llm.invoke(prompt)

    # Print content
    st.write(response.content)
