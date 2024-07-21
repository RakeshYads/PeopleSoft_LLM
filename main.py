# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from html_template import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_content = PdfReader(pdf)
        for page in pdf_content.pages:
            text += page.extract_text()
    return text

def get_chunk_text(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size= 1000,
        chunk_overlap=200,
        length_function= len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(chunk_text):
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vector_store = FAISS.from_text(texts = chunk_text, embeddings = embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(
        memory_key = "chat_history", return_message = True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm, retriever = vector_store.as_retriever(),
        memory = memory
    )
    return conversation_chain

def handle_user_input(user_question):

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
    # Use a breakpoint in the code line below to debug your script.
    load_dotenv()
    st.set_page_config(page_title="Chat With PeopleSoft Bot", page_icon= ":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        st.write("What!!")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat With PSoft Bot :books:")
    user_question = st.text_input("Ask a question about PeopleSoft to Bot")
    if user_question:
        handle_user_input(user_question)


    with st.sidebar:
        st.subheader("Uploaded Documnets")
        pdf_docs = st.file_uploader("Upload your files here and click on 'Process'", accept_multiple_files= True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text)

                #get the text chunks
                chunk_text = get_chunk_text(raw_text)
                #st.write(chunk_text)

                #create vector store
                vector_store = get_vector_store(chunk_text)

                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
