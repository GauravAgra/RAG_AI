import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

st.title("PDF RAG Agent with LangChain and Ollama")

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def create_vector_db(pdf_file):
    # Create a python PDF object
    pdf_reader = PdfReader(pdf_file)

    # Variable to store Documents
    documents = []

    for page_no, page in enumerate(pdf_reader.pages):
        text = page.extract_text()

        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"page": page_no + 1}
                )
            )

    # Initialize splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # Split content in chunks
    chunks = splitter.split_documents(documents)

    # Initialize embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Store chunks in a vector database
    vector_db = Chroma.from_documents(chunks, embeddings)

    return vector_db


# Load doc
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

# Upload doc
if pdf_file is not None:
    # Call vector db
    vector_db = create_vector_db(pdf_file)

    # Initialize retriever
    retriever = vector_db.as_retriever()

    # Initialize llm
    llm = ChatOllama(model="gemma3:1b", temperature=0)

    st.success("PDF indexed successfully")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input(
        "Ask a question about the PDF"
    )

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        prompt = f"""
    Answer the question using only the supplied context.

    Context:
    {context}

    Question:
    {question}
    """

        response = llm.invoke(prompt)

        answer = response.content

        with st.chat_message("assistant"):
            st.write(answer)

            with st.expander("Sources"):
                for doc in docs:
                    st.write(
                        f"Page {doc.metadata.get('page')}"
                    )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
