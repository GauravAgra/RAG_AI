from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

# Prepare doc

documents = [Document(page_content="""
Hello! My name is Gaurav Agrawal"""), Document(page_content="""
I work as senior software engineer in Citibank""")]

# Initialize embedding model

embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# Get Vector DB
vector_store = Chroma.from_documents(documents, embedding_model)

# Get question
question = "What is my name?"

# Prepare retriever
retriever = vector_store.as_retriever()

# Do similarity search
retrieved_doc = retriever.invoke(question)

# Convert retrieved doc into text
retrieved_text = "\n".join([doc.page_content for doc in retrieved_doc])
print(retrieved_text, end="\n\n")

# Prepare llmō
llm = ChatOllama(model="gemma3:270m")

# Prepare prompt
prompt = f"""
Using the below provided information, answer the question.

Information: {retrieved_text}

Question: {question}
"""

# Pass the result of similarity search and question into prompt and invoke llm
result = llm.invoke(prompt)

# Print result
print(result.content)
