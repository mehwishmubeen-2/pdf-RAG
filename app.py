import streamlit as st
import os

from dotenv import load_dotenv

from rag import (
    load_pdf,
    split_text,
    create_vectorstore,
    create_qa_chain
)

# LOAD ENV
load_dotenv()

# This looks locally first, then looks inside your Streamlit Advanced Secrets on the cloud!
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
# PAGE CONFIG
st.set_page_config(page_title="PDF RAG Chatbot")

st.title(" PDF RAG Chatbot using Groq")

# CHECK API KEY
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

# FILE UPLOAD
pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:

    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())

    st.success("PDF uploaded successfully!")

    # LOAD PDF
    pages = load_pdf("temp.pdf")

    # SPLIT
    docs = split_text(pages)

    # VECTORSTORE
    vectorstore = create_vectorstore(docs)

    # QA CHAIN
    qa_chain = create_qa_chain(
        vectorstore,
        GROQ_API_KEY
    )

    st.success("RAG system ready!")

    # USER QUESTION
    query = st.text_input("Ask question from PDF")

    if query:

        result = qa_chain.invoke({"query": query})

        st.subheader("Answer")
        st.write(result["result"])

        with st.expander("Source Chunks"):
            for doc in result["source_documents"]:
                st.write(doc.page_content)