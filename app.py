import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.cohere import CohereEmbeddings
import cohere
import weaviate
from langchain.vectorstores.weaviate import Weaviate
from langchain.vectorstores import Weaviate
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.chat_models import ChatCohere
from langchain.schema import HumanMessage
import asyncio
from langchain.retrievers import CohereRagRetriever
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


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
    vectorstore = Weaviate.from_texts(
        text_chunks, embeddings, client=client, by_text=False
    )
    return vectorstore

def get_conversation_chain(vectorstore):
    #loop = asyncio.new_event_loop()
    #retriever = loop.run_until_complete(get_retriever_async(vectorstore))

    llm = ChatCohere()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    st.set_page_config(page_title="Research InSight", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.header("Dive deeper into your researches :books:")
    user_question= st.text_input("Ask a question about your document")
    if user_question:
        handle_userinput(user_question)

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

            vectorstore = get_embeddings(text_chunks)
            st.success("Embeddings created successfully!")

            #vectorstore=vector_store(text_chunks, embeddings)
            #st.success("Embeddings stored successfully in Weaviate!")
            st.session_state.conversation = get_conversation_chain(
                vectorstore)


if __name__ == "__main__":
    main()