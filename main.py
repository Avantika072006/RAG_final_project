import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(PROJECT_DIR, "document_loader", "MIS Complete Notes.pdf")


def build_rag_app():
    if not os.path.isfile(PDF_PATH):
        raise FileNotFoundError(
            f"MIS notes PDF not found: {PDF_PATH}"
        )

    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NVDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set NVIDIA_API_KEY in the .env file before running the program."
        )

    print(f"Loading MIS notes from: {PDF_PATH}")
    documents = PyPDFLoader(PDF_PATH).load()
    print(f"Loaded {len(documents)} PDF pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    embeddings = FastEmbedEmbeddings()
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="mis_notes",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="mis_notes",
        embedding=embeddings,
    )
    vectorstore.add_documents(chunks)
    print("MIS notes indexed successfully.")

    llm = ChatNVIDIA(
        model=os.getenv(
            "NVIDIA_MODEL",
            "nvidia/nemotron-3.5-lightning-30b-a3b",
        ),
        nvidia_api_key=api_key,
        temperature=0.2,
        max_completion_tokens=2500,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert tutor for Management Information Systems (MIS). "
            "Answer the user's question using ONLY the information in the provided "
            "MIS notes context. Explain clearly, use headings or numbered steps when "
            "appropriate, and preserve the terminology used in the notes.\n"
            "Do not add facts that are not supported by the context. If the notes do "
            "not contain enough information to answer, say exactly: "
            "'I cannot find this information in the MIS notes.'\n\n"
            "MIS NOTES CONTEXT:\n{context}"
        ),
        ("human", "{question}"),
    ])

    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain, retriever


def query_rag(question: str, rag_chain, retriever):
    print("\n" + "="*60)
    print(f"Query: {question}")
    print("="*60)

    retrieved_docs = retriever.invoke(question)
    print(f"\nRetrieved {len(retrieved_docs)} relevant context chunk(s):")
    for i, doc in enumerate(retrieved_docs, start=1):
        raw_page = doc.metadata.get("page")
        page_num = raw_page + 1 if isinstance(raw_page, int) else "Unknown"
        preview = (
            doc.page_content.strip()
            .replace("\n", " ")
            .encode("ascii", errors="replace")
            .decode("ascii")[:150]
        )
        print(f" [{i}] (Page {page_num}): {preview}...")

    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', 0) + 1 if isinstance(doc.metadata.get('page'), int) else 'Unknown'}]\n"
        f"{doc.page_content}"
        for doc in retrieved_docs
    )

    print("\nGenerating MIS answer...")
    response = rag_chain.invoke({"context": context, "question": question})
    print("\nAnswer:")
    print(response.encode("ascii", errors="replace").decode("ascii"))
    print("="*60)
    return response


if __name__ == "__main__":
    rag_chain, retriever = build_rag_app()

    question = input(
        "\nAsk a question about the MIS notes "
        "(or press Enter for a sample question): "
    ).strip()
    if not question:
        question = "What is a Management Information System, and why is it important?"

    query_rag(question, rag_chain, retriever)
