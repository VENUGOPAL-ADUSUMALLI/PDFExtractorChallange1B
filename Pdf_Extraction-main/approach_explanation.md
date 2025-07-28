This project uses a Retrieval-Augmented Generation (RAG) approach to accurately answer questions by leveraging both structured inputs and unstructured data (PDFs). The workflow follows these core steps:

Input Parsing

The script takes a JSON file containing questions as input.

Each input may include additional metadata (e.g., categories or context).

PDF Document Ingestion

All PDFs in the specified folder are parsed using a PDF parser (e.g., PyMuPDF or pdfplumber).

Each document is split into smaller chunks (e.g., paragraphs or fixed-length spans).

Chunks are embedded into vector representations using a sentence embedding model (e.g., sentence-transformers).

Vector Store Creation

All embeddings are stored in a vector store (e.g., FAISS or Chroma) for efficient similarity search.

Retrieval

For each question, a semantic embedding is generated.

The top-k most relevant chunks from the PDF corpus are retrieved using vector similarity.

Prompt Construction

A prompt is dynamically created by combining:

The question from the input.

Retrieved PDF chunks as contextual information.

This helps the LLM answer accurately without hallucinating.

Answer Generation

The constructed prompt is passed to a large language model (e.g., OpenAI GPT-4 or a local LLM).

The model generates a concise and contextually grounded answer.

Output Writing

Each answer is saved in the required output JSON format, aligned with the original questions.