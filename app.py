import streamlit as st
from pdf_processor import PDFProcessor
from text_chunking import TextChunker
from embeddings_faiss import EmbeddingFAISS
from rag_qa import RAGQA
from summarizer import ResearchPaperSummarizer
from database import ResearchDatabase

st.title("Research Paper Summarizer using Generative AI (RAG)")

uploaded_file = st.file_uploader("Upload Research Paper PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    processor = PDFProcessor("temp.pdf")
    data = processor.preprocessing_pipeline()
    text = data["cleaned_text"]
    sentences = data["sentences"]

    st.write("Text extracted successfully.")

    chunker = TextChunker()
    chunks = chunker.chunk_by_words(text)

    st.write(f"Total Chunks Created: {len(chunks)}")

    vector_store = EmbeddingFAISS()
    vector_store.add_documents(chunks)

    summarizer = ResearchPaperSummarizer()
    summary = summarizer.summarize_document(chunks)

    st.subheader("Summary")
    st.write(summary)

    db = ResearchDatabase()
    paper_id = db.insert_paper("Uploaded Paper", summary)

    question = st.text_input("Ask a Question from Paper")

    if question:
        rag = RAGQA()
        context = vector_store.search(question)
        result = rag.answer_with_sources(context, question)

        st.subheader("Answer")
        st.write(result["answer"])

        db.insert_question(paper_id, question, result["answer"])