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


#IN THIS SECTION THE CHUNKING LOGIC WILL BE IMPLEMENTED. Here we are doing the chunking based on the new line, this can varry according to the documents
#ADDING EMBEDDINGS TO CHUNKS 
def add_embeddings(chunks):
    for chunk in chunks:
        vector = get_embedding(chunk["text"])
        chunk["vector"] = vector
         
    return chunks          # now we have the chunks with their embeddings, we can use them for searching and retrieval using the name chunks instead of documents.

#SEARCHING CHUNKS
def search_chunks(query, chunks, top_k=3):
    query_vector = get_embedding(query)
    results = []
    
    for chunk in chunks:
        score  = cosine_similarity(query_vector, chunk["vector"])   # finding the similarity btw the user querry and chunk stored embedding 
        results.append({
            "score": score,
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "size": len(chunk["text"])
        })
    
    # Sort by similarity in descending order and return the top_k results
    results.sort(key=lambda rec: rec["score"], reverse=True)  #sorting the similarity scores in descending order to get the most relevant documents first
    return results[:top_k]  #sending the top_k most relevant documents back to the user


#taking one file and creating multiple chunks based on the condition(here condition is based on the new line, but it can be changed according to the document type)
def chunk_document(document):
    chunks = []
    chunk_id = 0  # Initialize chunk ID counter

    for part in document["text"].split("\n\n"):    #going to the text part of the document and then splitting the text into multiple parts based on the new line
        part = part.strip()
        if part:
            chunks.append({
                "source": document["source"],
                "chunk_id": chunk_id,  # Assign a unique chunk ID
                "text": part
            })
            chunk_id += 1

    return chunks

#Creating chunks for all documents in the KB_FOLDER and returning a list of all chunks
def create_chunks():
    documents = load_documents(KB_FOLDER)
    chunks = []

    for doc in documents:
        chunks.extend(chunk_document(doc))       #here adding the chunks of each document to the main chunks list, so that we have all the chunks from all documents in one list

    return chunks
    
#TESTING METHOD 
def test(user_query, chunks, top_k=3):
    print(f"User Query: {user_query}")
    results = search_chunks(user_query, chunks, top_k)
   
    for r in enumerate(results):
        print(f"Source: {r[1]['source']}")
        print(f"Score: {r[1]['score']:.3f}")
        print(f"Size: {r[1]['size']} characters")
        print(f"Text: {r[1]['text'][:200]}...")  # Print first 200 characters of the text



#let's start 
chunks = create_chunks()  # Assuming you have a function to create chunks from documents
chunks = add_embeddings(chunks)

print(f"Loaded {len(chunks)} chunks with embeddings.")

user_query = input("Enter your query: ")
test(user_query, chunks, top_k=3)

