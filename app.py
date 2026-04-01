import streamlit as st
import tempfile
import os

from rag_utils import load_and_split, create_vector_store, create_qa_chain

st.set_page_config(page_title="RAG App", layout="wide")

st.title("📄 RAG Document Q&A (Python Output)")

# Session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Upload section
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    st.info("🔄 Processing document...")

    docs = load_and_split(file_path)
    db = create_vector_store(docs)
    qa_chain = create_qa_chain(db)

    st.session_state.qa_chain = qa_chain

    st.success("✅ Document processed successfully!")

# Question section
if st.session_state.qa_chain:
    st.subheader("Ask Questions")

    query = st.text_input("Enter your question")

    if query:
        with st.spinner("🤖 Thinking..."):
            response = st.session_state.qa_chain(query)

        st.code(response, language="python")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit + LangChain + FAISS")