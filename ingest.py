from pathlib import Path
from typing import List
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

load_dotenv()

DOCS_DIR = Path("data/docs")
PERSIST_DIR = "data/chroma_db"


def load_all_pdfs() -> List[Document]:
    docs: List[Document] = []

    for pdf_path in DOCS_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        file_docs = loader.load()

        for d in file_docs:
            d.metadata.setdefault("source", pdf_path.name)

        docs.extend(file_docs)

    return docs


def split_into_chunks(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    return splitter.split_documents(docs)


def build_vector_store(chunks: List[Document]) -> Chroma:
    embeddings = OpenAIEmbeddings()

    if os.path.exists(PERSIST_DIR):
        vector_store = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
        )
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
        )
        vector_store.persist()

    return vector_store


def ingest_corpus() -> Chroma:
    docs = load_all_pdfs()
    chunks = split_into_chunks(docs)
    return build_vector_store(chunks)
