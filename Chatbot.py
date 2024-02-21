

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from gtts import gTTS
from io import BytesIO

os.environ['GOOGLE_API_KEY'] = 'AIzaSyDeJfgMlMF1TJs_2DRRSn8FKfuUsDxDLWo'

def get_pdf_text(pdf_reader):
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    llm = GooglePalm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

def user_input(user_question, pdf_reader):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("You: ", message.content)
        else:
            st.write("CPGRAMS: ", message.content)
            tts = gTTS(message.content, lang='en')
            sound_file = BytesIO()
            tts.write_to_fp(sound_file)
            st.audio(sound_file, format="audio/mp3")

def main():
    st.set_page_config("CPGRAMS Chatbot")

    st.title("CPGRAMS Chatbot")
    user_question = st.text_input("Query About product")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None

    pdf_reader = PdfReader("CPGRAMS-Help.pdf")

    raw_text = get_pdf_text(pdf_reader)
    text_chunks = get_text_chunks(raw_text)
    vector_store = get_vector_store(text_chunks)
    st.session_state.conversation = get_conversational_chain(vector_store)
  

    if user_question:
        user_input(user_question, pdf_reader)

if __name__ == "__main__":
    main()




