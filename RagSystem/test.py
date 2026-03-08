from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = "./chroma_db"

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Loading existing Chroma DB...")

db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

print("Checking DB contents...")
count = db._collection.count()
print(f"✓ Total chunks in DB: {count}")

assert count > 0, "DB is empty or failed to load!"

# --------------------------------------------------
# TEST QUERY
# --------------------------------------------------

query = "What are the main clinical guidelines?"
print(f"\nQuery: {query}")

results = db.similarity_search_with_score(query, k=3)

print("\nTop Results:")
print("-" * 60)

for i, (doc, score) in enumerate(results, 1):
    print(f"\n[Result {i}]")
    print("Source:", doc.metadata.get("source"))
    print("Page:", doc.metadata.get("page"))
    print("Similarity Score:", round(score, 4))
    print("Content Preview:", doc.page_content[:300])
