import os

from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import numpy

# Get project root (simple-AI-assistance)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root
load_dotenv(PROJECT_ROOT / ".env")

# Read KB path from environment
kb_path = Path(os.getenv("KB_FOLDER", "./data"))

# Resolve relative paths from project root
KB_FOLDER = (
    kb_path if kb_path.is_absolute()
    else (PROJECT_ROOT / kb_path).resolve()
)

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

EMBED_MODEL = os.getenv("EMBEDDING_MODEL")

#embedding function to convert text into vector representation
def get_embedding(text):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

#cosine similarity function to measure the similarity between two vectors
def cosine_similarity(vec1, vec2):
    dot_prod = numpy.dot(vec1, vec2)
    norm1 = numpy.linalg.norm(vec1)
    norm2 = numpy.linalg.norm(vec2)
    return dot_prod / (norm1 * norm2) if norm1 and norm2 else 0.0

#LOADING DOCUMENTS
def load_documents(folder = KB_FOLDER):
    documents = []
    for filename in os.listdir(folder):               #"listdir" to get all the files in the specified folder
        if filename.endswith(".txt"):                      # safety check to ensure only text files are read
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:      #"with" to ensure the file is properly closed after reading
                documents.append({
                    "source": filename,
                    "text": f.read()
                })
    return documents

#ADDING EMBEDDINGS TO DOCUMENTS
def add_embeddings(documents):
    for doc in documents:
        vector = get_embedding(doc["text"])
        doc["vector"] = vector
         
    return documents

#SEARCHING DOCUMENTS
def search_documents(query, documents, top_k=3):
    query_vector = get_embedding(query)
    results = []
    
    for doc in documents:
        score  = cosine_similarity(query_vector, doc["vector"])   # finding the similarity btw the user querry and document stored embedding 
        results.append({
            "score": score,
            "source": doc["source"],
            "text": doc["text"],
            "size": len(doc["text"])
        })
    
    # Sort by similarity in descending order and return the top_k results
    results.sort(key=lambda rec: rec["score"], reverse=True)  #sorting the similarity scores in descending order to get the most relevant documents first
    return results[:top_k]  #sending the top_k most relevant documents back to the user

#TESTING METHOD 
def test(user_query, documents, top_k=3):
    print(f"User Query: {user_query}")
    results = search_documents(user_query, documents, top_k)
   
    for r in enumerate(results):
        print(f"Source: {r[1]['source']}")
        print(f"Score: {r[1]['score']:.3f}")
        print(f"Size: {r[1]['size']} characters")
        print(f"Text: {r[1]['text'][:200]}...")  # Print first 200 characters of the text


#let's start 
documents = load_documents(KB_FOLDER)
documents = add_embeddings(documents)

print(f"Loaded {len(documents)} documents with embeddings.")

user_query = input("Enter your query: ")
test(user_query, documents, top_k=3)

