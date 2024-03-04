#document_loaders
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

# from decouple import config

# set a flag to switch between local and remote parsing
# change this to True if you want to use local parsing
local = False

# Directory to save audio files
save_dir = "./YouTube"

# set openai api key
# os.environ['OPENAI_API_KEY'] = config("OPENAI_API_KEY")

try:
    # get url
    url = ["https://youtube.com/shorts/IicbiwTAslE?si=H1qA7---M4ZiuHTc"]
    # Transcribe the videos to text
    if local:
        loader = GenericLoader(YoutubeAudioLoader(url, save_dir), OpenAIWhisperParserLocal())
    else:
        loader = GenericLoader(YoutubeAudioLoader(url, save_dir), OpenAIWhisperParser())
    docs = loader.load()

    # Combine docs
    combined_docs = [doc.page_content for doc in docs]
    text = " ".join(combined_docs)
    print(text)

    # Split the combined docs into chunks of size 1500 with an overlap of 150
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_text(text)

    # Build an index
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(splits, embedding_function,
                                 persist_directory="vector_store_0003",
                                 collection_name="david_goggins_short")

except Exception as e:
    print(e)