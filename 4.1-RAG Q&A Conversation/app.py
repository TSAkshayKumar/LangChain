## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory # provide in memory storage
from langchain_core.chat_history import BaseChatMessageHistory # provide structure
from langchain_core.runnables.history import RunnableWithMessageHistory # easy to use in langchain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

# os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


## set up Streamlit 
st.title("Conversational RAG With PDF uplaods and chat history")
st.write("Upload Pdf's and chat with their content")

## Input the Groq API Key
api_key=st.text_input("Enter your Groq API key:",type="password")

## Check if groq api key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")

    ## chat interface

    session_id=st.text_input("Session ID",value="default_session") #capture session Id
    
    ## statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
    ## Process uploaded  PDF's
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files: #for each files
            temppdf=f"./temp.pdf"  #temp path
            with open(temppdf,"wb") as file: # if file exist then open it otherwise create 
                file.write(uploaded_file.getvalue()) # write content to file
                file_name=uploaded_file.name # name file stored in var [no use]
            # since we are uploading the file that is why we need to store it at some location as pdf
            # PyPDFLoader need pdf file path
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs) # adding docs to documents list 

    # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()    

        #Prompt for how to use context history 
        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        ) # create standalone question
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        #instance
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        # Answer question
        system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
        qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain) #instance Only QA with place to add history message

    # define function that stores message history. It inherit BaseChatMessageHistory. ChatMessageHistory() store all messages for a session.
        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory() #instance of storage
            return st.session_state.store[session_id] #return storage instance
        
        # create runnable instance. [Q&A with history message]
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain, # rag chain for QA
            get_session_history, #function to provide storage
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                }, 
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            session_history=get_session_history(session_id)
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the GRoq API Key")










