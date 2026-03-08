import os
import torch
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# PDF processing
import PyPDF2

# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


class RAGSystem:
    """
    A Retrieval-Augmented Generation system for PDF documents.
    Uses BioBERT embeddings and ChromaDB for vector storage.
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG system with BioBERT embeddings.
        
        Args:
            persist_directory: Directory to persist ChromaDB
        """
        print("Initializing RAG System...")
        
        # Initialize BioBERT embeddings
        print("Loading BioBERT embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
            model_kwargs={
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            },
            encode_kwargs={
                "batch_size": 16,
                "normalize_embeddings": True
            }
        )
        
        self.persist_directory = persist_directory
        self.vectorstore = None
        
        print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[Document]:
        """
        Extract text from a PDF file and create Document objects.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects with text and metadata
        """
        documents = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                filename = os.path.basename(pdf_path)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    
                    if text.strip():  # Only add non-empty pages
                        doc = Document(
                            page_content=text,
                            metadata={
                                'source': filename,
                                'page': page_num + 1,
                                'total_pages': len(pdf_reader.pages)
                            }
                        )
                        documents.append(doc)
                        
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            
        return documents
    
    def process_pdfs(self, pdf_folder: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Process all PDFs in a folder and create ChromaDB vector store.
        
        Args:
            pdf_folder: Path to folder containing PDF files
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        # Convert to Path object and resolve absolute path
        pdf_folder_path = Path(pdf_folder).resolve()
        
        print(f"\nLooking for PDFs in: {pdf_folder_path}")
        
        if not pdf_folder_path.exists():
            print(f"❌ Error: Folder '{pdf_folder_path}' does not exist!")
            print(f"\nCurrent directory: {Path.cwd()}")
            return
        
        pdf_files = list(pdf_folder_path.glob('*.pdf'))
        
        if not pdf_files:
            print(f"❌ No PDF files found in {pdf_folder_path}")
            print(f"\nPlease check:")
            print(f"1. The folder path is correct")
            print(f"2. PDF files exist in the folder")
            print(f"3. Files have .pdf extension")
            return
        
        print(f"✓ Found {len(pdf_files)} PDF files")
        
        # Extract text from all PDFs
        all_documents = []
        print("\nExtracting text from PDFs...")
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            docs = self.extract_text_from_pdf(str(pdf_path))
            all_documents.extend(docs)
        
        print(f"✓ Extracted {len(all_documents)} pages from PDFs")
        
        # Split documents into chunks
        print("\nSplitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(all_documents)
        print(f"✓ Created {len(chunks)} text chunks")
        
        # Create ChromaDB vector store
        print("\nCreating embeddings and storing in ChromaDB...")
        print("⏳ This may take a few minutes depending on the number of chunks...")
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"\n{'='*60}")
        print(f"✓ Vector store created successfully!")
        print(f"✓ Stored {len(chunks)} chunks in ChromaDB")
        print(f"✓ Database persisted to: {self.persist_directory}")
        print(f"{'='*60}")
        
    def load_existing_vectorstore(self):
        """
        Load an existing ChromaDB vector store from disk.
        """
        if not os.path.exists(self.persist_directory):
            print(f"No existing vector store found at {self.persist_directory}")
            return False
        
        print(f"Loading existing vector store from {self.persist_directory}...")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        print("✓ Vector store loaded successfully!")
        return True
        
    def query(self, question: str, k: int = 5) -> List[Dict]:
        """
        Query the RAG system and retrieve relevant documents.
        
        Args:
            question: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with metadata and similarity scores
        """
        if self.vectorstore is None:
            print("Error: Vector store not initialized. Please process PDFs first or load existing store.")
            return []
        
        # Perform similarity search with scores
        results = self.vectorstore.similarity_search_with_score(question, k=k)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': float(score)
            })
        
        return formatted_results
    
    def query_with_context(self, question: str, k: int = 5) -> str:
        """
        Query the RAG system and return formatted context for LLM.
        
        Args:
            question: Query string
            k: Number of documents to retrieve
            
        Returns:
            Formatted context string
        """
        results = self.query(question, k=k)
        
        if not results:
            return "No relevant documents found."
        
        context = "Retrieved Context:\n\n"
        for i, result in enumerate(results, 1):
            context += f"[Document {i}]\n"
            context += f"Source: {result['metadata']['source']} (Page {result['metadata']['page']})\n"
            context += f"Content: {result['content']}\n"
            context += f"Relevance Score: {result['similarity_score']:.4f}\n\n"
        
        return context
    
    def delete_vectorstore(self):
        """
        Delete the persisted vector store.
        """
        import shutil
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print(f"✓ Vector store deleted from {self.persist_directory}")
            self.vectorstore = None


def main():
    """
    Main function to demonstrate the RAG system usage.
    """
    print("="*60)
    print("RAG SYSTEM FOR PDF GUIDELINES")
    print("="*60)
    
    # Get PDF folder path from user or use default
    print("\nCurrent directory:", Path.cwd())
    
    # Try to find GUIDELINES folder
    possible_paths = [
        "GUIDELINES",
        "./GUIDELINES",
        "../GUIDELINES",
        "D:\GUIDELINES_mod\GUIDELINES_mod\GUIDELINES"
    ]
    
    pdf_folder = None
    for path in possible_paths:
        if Path(path).exists():
            pdf_files = list(Path(path).glob('*.pdf'))
            if pdf_files:
                pdf_folder = path
                print(f"\n✓ Found GUIDELINES folder at: {Path(path).resolve()}")
                print(f"✓ Contains {len(pdf_files)} PDF files")
                break
    
    if pdf_folder is None:
        print("\n❌ Could not automatically find GUIDELINES folder with PDFs")
        pdf_folder = input("\nEnter the full path to your GUIDELINES folder: ").strip()
        
        if not pdf_folder:
            print("No path provided. Using default 'GUIDELINES'")
            pdf_folder = "GUIDELINES"
    
    # Initialize RAG system
    rag = RAGSystem(persist_directory="./chroma_db")
    
    # Check if vector store already exists
    if not rag.load_existing_vectorstore():
        # Process PDFs and create vector store
        print("\n" + "="*60)
        print("PROCESSING PDFs")
        print("="*60)
        rag.process_pdfs(
            pdf_folder=pdf_folder,
            chunk_size=1000,
            chunk_overlap=200
        )
    
    # Only run example queries if vector store exists
    if rag.vectorstore is not None:
        # Example queries
        print("\n" + "="*60)
        print("RAG SYSTEM READY!")
        print("="*60)
        
        # Example query
        example_query = "What are the main guidelines?"
        print(f"\nExample Query: '{example_query}'")
        print("\nTop 3 relevant chunks:")
        print("-"*60)
        
        results = rag.query(example_query, k=3)
        for i, result in enumerate(results, 1):
            print(f"\n[Result {i}]")
            print(f"Source: {result['metadata']['source']} (Page {result['metadata']['page']})")
            print(f"Similarity Score: {result['similarity_score']:.4f}")
            print(f"Content Preview: {result['content'][:200]}...")
            print("-"*60)
    else:
        print("\n❌ Vector store could not be created. Please check the folder path and try again.")


if __name__ == "__main__":
    main()