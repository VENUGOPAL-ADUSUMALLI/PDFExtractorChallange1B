# Adobe Hack - RAG Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline to solve Adobe Hack challenges using a combination of PDFs and structured input JSONs. The core logic processes inputs and references the relevant documents to generate answers using an LLM.

---

## Project Structure

```
Adobe_Hack/
│
├── 1b/                         # Challenge input/output folder
│   ├── Collection 1/
│   ├── Collection 2/
│   └── Collection 3/
│       ├── challenge1b_input.json
│       ├── challenge1b_output.json
│       └── PDFs/
│           └── All reference documents (PDFs)
│
├── ragpipeline/               # Main pipeline code
│   ├── main.py                # Entry point to run the pipeline
│   ├── utils.py               # Helper functions for file I/O, PDF parsing, etc.
│   ├── config.py              # Configuration file (e.g., LLM settings)
│   └── test.py                # Test script to validate outputs
│
├── venv/                      # Virtual environment
│
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── .gitignore
```

---

## How to Run (Locally)

### 📌 Prerequisites

1. Python 3.8+
2. Virtual environment activated (recommended)
3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Run the pipeline

To run the pipeline on **Collection 3** under `1b`, use:

```bash
python3 ragpipeline/main.py "1b/Collection 3/challenge1b_input.json" "1b/Collection 3/PDFs"
```

Replace the input path and PDFs folder with those of other collections as needed.

---

## Run with Docker

### Step 1: Build Docker Image

```bash
docker build -t pdf_extraction-app .
```

### Step 2: Run the Container

```bash
docker run -d --name pdfextractor pdf_extraction-app
```

This runs the pipeline on `Collection 3` as defined in the Dockerfile's default command.

> To change the input collection, modify the `CMD` instruction in the Dockerfile or override it in the `docker run` command.

###  Step 3 (Optional): Push to Docker Hub

```bash
docker tag pdfextractor-cli venugopal376/pdf_extraction-app
docker login
docker push venugopal376/pdf_extraction-app
```

---

## Output

The script will generate responses for each question in the input JSON, using relevant content from the PDFs, and save the results in:

```
1b/Collection 3/challenge1b_output.json
```

---

## Example Collections

* **Collection 1**: Travel Planning
* **Collection 2**: Management Creation
* **Collection 3**: Menu Planning

Each contains:

* `challenge1b_input.json` – the input questions
* `PDFs/` – reference material
* `challenge1b_output.json` – generated answers

---

