import os
import glob
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

# 2. Document Loading
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_report():
    for pattern in (
        "activity_1_decision_tree_report.pdf",
        "Activity_1_Decision_Tree_Report.pdf",
        "activity_1_decision_tree_report.docx",
        "Activity_1_Decision_Tree_Report.docx",
        "*Activity*Decision*Tree*Report*.pdf",
        "*activity*decision*tree*report*.pdf",
        "*Activity*Decision*Tree*Report*.docx",
        "*activity*decision*tree*report*.docx",
    ):
        matches = glob.glob(os.path.join(PROJECT_DIR, pattern))
        if matches:
            return matches[0]

    supported_files = (
        glob.glob(os.path.join(PROJECT_DIR, "*.pdf"))
        + glob.glob(os.path.join(PROJECT_DIR, "*.docx"))
    )
    if supported_files:
        return supported_files[0]

    raise FileNotFoundError(
        "No PDF or DOCX report found in the project folder. "
        "Place Activity_1_Decision_Tree_Report.pdf or "
        "Activity_1_Decision_Tree_Report.docx beside main.py."
    )


def load_report(report_path: str) -> list[Document]:
    extension = os.path.splitext(report_path)[1].lower()
    source_name = os.path.basename(report_path)

    if extension == ".pdf":
        documents = PyPDFLoader(report_path).load()
    elif extension == ".docx":
        doc = DocxDocument(report_path)
        sections = []

        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                sections.append(text)

        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip().replace("\n", " ")
                    for cell in row.cells
                ).strip(" |")
                if row_text:
                    sections.append(row_text)

        if not sections:
            raise ValueError(f"No readable text found in: {report_path}")

        documents = [Document(
            page_content="\n".join(sections),
            metadata={"source": report_path, "source_name": source_name},
        )]
    else:
        raise ValueError(f"Unsupported report format: {extension}")

    for document in documents:
        document.metadata["source"] = report_path
        document.metadata["source_name"] = source_name
    return documents


report_path = find_report()
if not api_key:
    raise EnvironmentError("NVIDIA_API_KEY is not set in the .env file.")

print(f"Loading report from: {report_path}")
docs = load_report(report_path)
print(f"Loaded {len(docs)} document section(s).")

# 3. Document Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
)
chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} text chunks.")

# 4. Embeddings & In-Memory Retrieval
print("Creating FastEmbed embeddings and in-memory index...")
embeddings = FastEmbedEmbeddings()
chunk_embeddings = np.asarray(
    embeddings.embed_documents([chunk.page_content for chunk in chunks]),
    dtype=np.float32,
)
chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)
chunk_norms = np.where(chunk_norms == 0, 1.0, chunk_norms)
print("Indexing complete!")

# 5. Initialize NVIDIA LLM
# Note: nemotron uses reasoning tokens, so max_completion_tokens should be >= 1024
llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    nvidia_api_key=api_key,
    temperature=0.2,
    max_completion_tokens=2500,
)

# 6. Retriever & RAG Prompt
def retrieve_documents(question: str, k: int = 3):
    query_embedding = np.asarray(
        embeddings.embed_query(question),
        dtype=np.float32,
    )
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        query_norm = 1.0

    similarities = (
        chunk_embeddings @ query_embedding
        / (chunk_norms * query_norm)
    )
    top_indices = np.argsort(similarities)[::-1][:k]
    return [chunks[index] for index in top_indices]

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an academic assistant for the Activity 1 Decision Tree report. "
        "Answer accurately using ONLY the provided report context. "
        "Preserve the report's terminology and explain the decision-tree concepts clearly. "
        "If the answer cannot be found in the report, say that you don't know.\n\n"
        "Context:\n{context}"
    ),
    ("human", "{question}"),
])

rag_chain = prompt | llm | StrOutputParser()

# 7. Test RAG with a question
def query_rag(question: str):
    print("\n" + "="*60)
    print(f"Query: {question}")
    print("="*60)
    
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(question)
    print(f"\nRetrieved {len(retrieved_docs)} relevant context chunk(s):")
    for i, doc in enumerate(retrieved_docs, start=1):
        page_num = doc.metadata.get("page", "Unknown")
        preview = doc.page_content.strip().replace("\n", " ")[:150]
        print(f" [{i}] (Page {page_num}): {preview}...")

    # Build context string
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Generate response
    print("\nGenerating response from NVIDIA LLM...")
    response = rag_chain.invoke({"context": context, "question": question})
    print("\nAnswer:")
    print(response)
    print("="*60)
    return response

if __name__ == "__main__":
    print("\nAsk a question about the Activity 1 Decision Tree report.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            query_rag(question)