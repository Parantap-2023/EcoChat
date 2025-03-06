import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Set up embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to extract text from Excel files
def extract_text_from_excel(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            filepath = os.path.join(directory, filename)
            xls = pd.ExcelFile(filepath)
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)
                text = df.to_string(index=False)  # Convert DataFrame to string
                documents.append(Document(page_content=text, metadata={"source": filename, "sheet": sheet_name}))
    return documents

# Load Excel files and create embeddings
excel_docs = extract_text_from_excel("excel_files")  # Ensure this folder exists

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(excel_docs)

# Create FAISS vector store
vector_store = FAISS.from_documents(final_documents, embeddings)

# Save vector store for later use
vector_store.save_local("faiss_index")

print("✅ Embeddings created and saved successfully!")
