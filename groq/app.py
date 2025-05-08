import os
import time
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings # Embedding model - for text representation

def response_generator(response: str):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]

# Load the FAISS index from the prebuilt store
STORE_DIR = Path(__file__).parent / "faiss_store"

# load prebuilt FAISS index only once per session
@st.cache_resource
def get_vectorstore() -> FAISS:
    embeddings = OllamaEmbeddings(model="llama3.2")
    # only if we trust the source of the data
    return FAISS.load_local(STORE_DIR, embeddings, allow_dangerous_deserialization=True)

# Check if the vectorstore is already in session state
# If not, create it and store it in session state
if "vectors" not in st.session_state:
    st.session_state.vectors = get_vectorstore() # Load the vectorstore into session state

llm = ChatGroq(
    groq_api_key=groq_api_key, 
    model="gemma2-9b-it",
    temperature=0.3)

prompt_template = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant that answers questions based on provided context ONLY.
    Provide the most accurate response based on the question.

    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Create the document chain and retrieval chain
doc_chain = create_stuff_documents_chain(llm, prompt_template) # input all the documents to the LLM
retriever = st.session_state.vectors.as_retriever()
qa_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=doc_chain) # pipeline to retrieve the documents and generate the answer


st.title("LangChain Groq Chat")

if "history" not in st.session_state:
    st.session_state.history = []
    
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

if user_input := st.chat_input("Ask me anything about the page…"):
    # display user
    st.session_state.history.append(("user", user_input))
    st.chat_message("user").write(user_input)
    
    # retrieve & generate
    start = time.process_time()
    result = qa_chain.invoke({"input": user_input}) # ask the LLM a question based on the documents
    answer = result["answer"]
    
    # assistant reply
    st.session_state.history.append(("assistant", answer))
    with st.chat_message("assistant"):
        placeholder = st.empty()
        text_accum = ""
        for chunk in response_generator(answer):
            text_accum += chunk
            placeholder.write(text_accum)
    st.write(f"_Response time: {time.process_time() - start:.2f}s_")

    # show sources
    with st.expander("Document Sources"):
        for i, doc in enumerate(result["context"]):
            st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'n/a')}")
            st.write(doc.page_content)
            st.write("---")