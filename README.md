# Finance RAG for Quarterly Reports

A Retrieval-Augmented Generation application for answering questions from quarterly financial report PDFs. The system runs locally using Ollama for embeddings and language generation, ChromaDB for vector storage, and Streamlit for the user interface.

## Overview

This project allows users to upload quarterly financial report PDFs for a listed company and ask natural-language questions about revenue, profit, margins, dividends, risks, management commentary, and other financial details.

The application extracts text from uploaded PDFs, splits it into searchable chunks, stores embeddings in a persistent vector database, retrieves relevant context for each question, and generates grounded answers with source file and page references.

No OpenAI, Google, or other external LLM API keys are required.

## Features

- Upload multiple quarterly financial report PDFs
- Extract text page by page from selectable PDFs
- Preserve source metadata including file name, quarter, and page number
- Chunk financial text with overlap for better retrieval
- Generate embeddings locally using Ollama
- Store document chunks in persistent ChromaDB
- Retrieve relevant chunks for user questions
- Generate answers using a local Ollama chat model
- Display citations with source PDF and page number
- Refuse questions when the answer is not available in the uploaded reports
- Streamlit interface with upload, indexing, question answering, and answer history

## Tech Stack

- Python
- Streamlit
- ChromaDB
- pdfplumber
- Ollama

## Models Used

| Purpose | Model |
|---|---|
| Embeddings | `nomic-embed-text` |
| Answer generation | `llama3.2:3b` or `llama3.1:8b` |

`llama3.2:3b` is recommended for laptops with limited memory. `llama3.1:8b` can be used on systems with more RAM or better hardware.

## Project Structure

```text
finance-rag-ollama/
├── app.py
├── requirements.txt
├── README.md
├── EXECUTION_PLAN.md
├── TEST_QUESTIONS.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── chunking.py
│   ├── config.py
│   ├── ollama_client.py
│   ├── pdf_loader.py
│   ├── rag.py
│   └── vector_store.py
└── data/
    └── pdfs/
        └── .gitkeep

