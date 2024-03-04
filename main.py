import os, chromadb
import streamlit as st
from decouple import config         # pip install python-decouple

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, WebBaseLoader, PyPDFDirectoryLoader, CSVLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser, OpenAIWhisperParserLocal
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

from langchain_openai import ChatOpenAI
from langchain_community.llms       import LlamaCpp, CTransformers
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import LlamaCppEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS

#langchain root
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.prompts       import PromptTemplate
from langchain.chains        import LLMChain, RetrievalQA
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# emoji source: https://emojipedia.org/open-book
st.set_page_config(page_title="ChatGPT Clone",page_icon="🤖", layout="centered")

st.title("Chat With YouTube Video")

# set openai api key
os.environ['OPENAI_API_KEY'] = config("OPENAI_API_KEY")

persistent_client = chromadb.PersistentClient(path="./vector_store_0003")

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma(
    persist_directory="./vector_store_0003",
    collection_name="david_goggins_short",
    embedding_function=embedding_function,
)

prompt = PromptTemplate(
    template="""Given the context about a video. Answer the user in a friendly and precise manner.
    
    Context: {context}
    
    Human: {question}
    
    AI:""",
    input_variables=["context", "question"]
)

# Build a QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo",
                   temperature=0),
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": prompt
    }
)


if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there, am Prince"}]


# Display chat messages to screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# get user prompt
user_prompt = st.chat_input()

# check is user typed in something
if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)


# Get the last message, if not AI(assistant), generate LLM response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response = qa_chain(
                {"query": user_prompt})
            st.write(ai_response["result"])
            # st.write(ai_response["source_documents"])
            for doc in ai_response["source_documents"]:
                st.write(doc)
    new_ai_message = {"role": "assistant", "content": ai_response}
    st.session_state.messages.append(new_ai_message)
           