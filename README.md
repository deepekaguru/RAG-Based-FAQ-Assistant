# RAG-Based FAQ Assistant

A Retrieval-Augmented Generation (RAG) system that answers domain-specific questions from a document knowledge base using semantic search and LLM generation.

## Overview

Traditional FAQ systems rely on keyword matching and break down when users ask questions in natural language. This assistant uses RAG to retrieve the most relevant document chunks semantically and generate accurate, context-aware answers using a large language model.

## How It Works

1. **Document Ingestion** — Load and chunk source documents (PDF, TXT, or web content)
2. **Embedding** — Convert chunks into vector embeddings using OpenAI Embeddings
3. **Storage** — Store embeddings in ChromaDB vector store
4. **Retrieval** — On user query, retrieve top-k semantically similar chunks
5. **Generation** — Pass retrieved context + query to LLM to generate a grounded answer

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | OpenAI GPT-4o / GPT-3.5-turbo |
| Embeddings | OpenAI text-embedding-ada-002 |
| Vector Store | ChromaDB |
| Orchestration | LangChain |
| UI | Streamlit |
| Language | Python 3.10+ |

## Features

- Natural language question answering over custom documents
- Semantic similarity search — not keyword matching
- Source citation — shows which document chunk the answer came from
- Configurable chunk size and retrieval top-k
- Clean Streamlit UI for interactive querying
- Supports PDF, TXT, and plain text document formats

## Project Structure
