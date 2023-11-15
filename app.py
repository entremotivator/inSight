import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_pdf_text(pdf_papers):
    text = ""
    for pdf in pdf_papers:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    print(text[:500])
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1024,
        chunk_overlap=256,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk[:100]}...")
    return chunks


def main():
    load_dotenv()
    st.set_page_config(page_title="Research InSight", page_icon=":books:")
    st.header("Dive deeper into your researches :books:")
    st.text_input("Ask a question about your document")

    with st.sidebar:
        st.subheader("Your Research PDFs")
        pdf_papers = st.file_uploader(
            "Upload your PDFs here!",
            accept_multiple_files=True)
        if st.button("Process"):
            st.spinner("Getting InSight...")

            raw_text = get_pdf_text(pdf_papers)
            st.write(raw_text)

            text_chunks = get_text_chunks(raw_text)


if __name__ == "__main__":
    main()
