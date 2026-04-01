import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline


# 🔹 Load + Split
def load_and_split(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    return splitter.split_documents(documents)


# 🔹 Embeddings (HuggingFace)
def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.from_documents(docs, embeddings)
    return db


# 🔹 LLM (HuggingFace)
def load_llm():
    pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


# 🔹 QA Chain
def create_qa_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = load_llm()

    def ask_question(query):
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
            You are a helpful assistant.

            Answer ONLY from context.
            If not found, say "I don't know".

            Return STRICT Python dictionary:

            {{
                "answer": "...",
                "source_summary": "...",
                "confidence": "high/medium/low"
            }}

            Context:
            {context}

            Question:
            {query}
        """

        return llm.invoke(prompt)

    return ask_question