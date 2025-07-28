FROM python:3.11-slim

WORKDIR /pdf_Extraction-main
COPY requirements.txt .
RUN pip install -r requirements.txt


COPY . .
# Set the default command to run your main script
CMD ["python", "ragpipeline/main.py", "1b/Collection 3/challenge1b_input.json", "1b/Collection 3/PDFs"]
