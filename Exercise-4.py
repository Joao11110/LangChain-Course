from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def us_contitucion_helper(question):
    loader = TextLoader("US_constitution.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=400)
    docs = text_splitter.split_documents(documents)

    embbed_function = OpenAIEmbeddings()
    db = Chroma.from_documents(docs, embbed_function, persist_directory="/result")
    db.persist()

    chat = ChatOpenAI(temperature=0)
    compressor = LLMChainExtractor.from_llm(chat)

    compressor_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriaver=db.as_retriever())
    compressor_docs = compressor_retriever.get_relevant_documents(question)

    print(compressor_docs[0].page_content)
