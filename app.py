import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.cohere import CohereEmbeddings
import cohere
import weaviate
from langchain.vectorstores.weaviate import Weaviate
import os

load_dotenv()
YOUR_COHERE_KEY = os.getenv('COHERE_API_KEY')
YOUR_WEAVIATE_KEY = os.getenv('WEAVIATE_API_KEY')
YOUR_WEAVIATE_URL = os.getenv('WEAVIATE_URL')
co = cohere.Client(YOUR_COHERE_KEY)


client = weaviate.Client(
    url='https://insight-enkxkddw.weaviate.network',  # Replace with your endpoint
    auth_client_secret=weaviate.AuthApiKey(api_key=YOUR_WEAVIATE_KEY),
    additional_headers={
        "X-Cohere-Api-Key": YOUR_COHERE_KEY # Replace with your inference API key
    }
)
client.schema.delete_all()
client.schema.get()
schema = {
    "classes":[
        {
            "class": "InSightChatbot",
            "description": "Documents for chatbot",
            "vectorizer": "text2vec-cohere",
            "moduleConfig": {
                "text2vec-cohere": {"model": "embed-english-light-v3.0", "truncate": "RIGHT"},
            },
            "properties": [
                {
                    "dataType": ["text"],
                    "description": "The content of the paragraph",
                    "moduleConfig": {
                        "text2vec-cohere": {
                            "skip": False,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "content",
                },
            ],
        }
    ]
}

client.schema.create(schema)
# vectorstore = Weaviate(client, "InSightChatbot", "content", attributes=["source"])


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


def get_embeddings(text_chunks):
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key=YOUR_COHERE_KEY)
    res = embeddings.aembed_query(text_chunks)
    return res


def vector_store(text_chunks, my_embeddings):
    vectorstore = Weaviate.afrom_texts(
        text_chunks, my_embeddings, client=client, by_text=False
    )



def main():
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
            st.success("Text extracted successfully!")

            text_chunks = get_text_chunks(raw_text)
            st.success("Chunks created successfully!")

            embeddings = get_embeddings(text_chunks)
            st.success("Embeddings created successfully!")

            vector_store(text_chunks, embeddings)
            st.success("Embeddings stored successfully in Weaviate!")


if __name__ == "__main__":
    main()