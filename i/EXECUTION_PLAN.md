# Execution Plan

Deadline: today, 5 PM

## Goal

Build and submit a GitHub repository for a local Finance RAG app that answers questions from quarterly financial report PDFs using Ollama, ChromaDB, and Streamlit. No OpenAI or Google API keys are used.

## Implementation Plan

1. Project setup
   - Create Python project structure.
   - Add `.gitignore` for virtual environment, local PDFs, and generated Chroma database.
   - Add `requirements.txt`.

2. Local model setup
   - Install Ollama.
   - Pull `nomic-embed-text` for embeddings.
   - Pull `llama3.1:8b` for answers.

3. PDF ingestion
   - Upload 3-4 quarterly PDFs.
   - Extract selectable text page by page.
   - Preserve file name, page number, and quarter metadata.

4. Chunking
   - Use 1200 character chunks.
   - Use 180 character overlap.
   - Prefix chunk text with file, quarter, and page so retrieval can distinguish similar quarters.

5. Embedding and storage
   - Embed chunks with Ollama.
   - Save embeddings, text, and metadata in persistent ChromaDB.
   - Use stable chunk IDs to avoid duplicate chunks on repeated indexing.

6. Retrieval and answering
   - Embed the user question with the same embedding model.
   - Retrieve the top relevant chunks from Chroma.
   - Ask the local chat model to answer only from retrieved context.
   - Refuse questions that cannot be answered from the reports.

7. Interface
   - Build Streamlit upload, index, question, answer, source, and history views.
   - Show progress messages during indexing and answering.
   - Disable asking until documents are indexed.

8. Submission
   - Fill README source PDF register.
   - Run the 10 assignment questions.
   - Add screenshots.
   - Record the short demo video.
   - Push the repository to GitHub and submit the repo link.

## Priority

The required app comes first. The optional FastAPI backend should only be attempted after the Streamlit app, README, screenshots, and test answers are complete.

