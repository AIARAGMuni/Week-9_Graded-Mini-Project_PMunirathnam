# Mini-RAG for Sherlock Holmes Stories

## Project Overview

This project implements a simple Retrieval-Augmented Generation (RAG) system using a collection of Sherlock Holmes stories. The system loads documents, splits them into overlapping chunks, generates embeddings using OpenAI, stores embeddings in Qdrant, retrieves relevant chunks for a user query, and uses an LLM to generate grounded answers with citations.

The goal of the project is to demonstrate the complete RAG workflow:

1. Document Ingestion
2. Text Chunking
3. Embedding Generation
4. Vector Storage
5. Semantic Retrieval
6. Grounded Answer Generation
7. Validation and Evaluation

## Setup

### Install Dependencies

pip install openai qdrant-client python-dotenv

### Configure Environment

Create a '.env' file:

OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key


## Usage

### Ingest Documents

python mp2_rag.py ingest

### Ask Questions

python mp2_rag.py ask

### Run Validation

python mp2_rag.py validate


## Learning Outcomes

Through this project I gained hands-on experience with:

- Retrieval-Augmented Generation (RAG)
- OpenAI Embeddings
- Qdrant Vector Database
- Semantic Search
- Prompt Engineering
- Context Grounding
- Evaluation of RAG Systems
- End-to-End LLM Application Development

The project demonstrated that successful RAG systems depend on the combination of chunking strategy, retrieval quality, prompt design, and evaluation, not just the language model itself.
