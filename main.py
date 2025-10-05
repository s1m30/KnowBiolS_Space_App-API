# main.py

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import sqlite3
import json
from neo4j import GraphDatabase
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv
import os 
# --- 1. Data Models (using Pydantic) ---
# Pydantic is used for request/response data validation and serialization.

load_dotenv()
NEO4J_KEY = os.getenv("NEO4j_key")
class Paper(BaseModel):
    """
    Schema for an individual item in our inventory.
    id is optional upon creation but required when returning the item.
    """
    id: Optional[int] = Field(None, description="The unique ID of the item.")
    title: str = Field(..., description="The name of the item.")


# --- 2. Application Setup ---
app = FastAPI(
    title="Knowledge Biology Engine Backend",
    description="A simple API demonstrating CORS and Pydantic validation."
)


# In a real-world scenario, this would connect to a database.
# For simplicity, we use an in-memory dictionary.
# fake_db = {
#     1: {"name": "Laptop", "price": 1200.00, "is_available": True},
#     2: {"name": "Mouse", "price": 25.50, "is_available": False},
#     3: {"name": "Keyboard", "price": 75.99, "is_available": True},
# }
# next_id = 4


DB_PATH = "chroma.sqlite3"
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # returns dict-like rows
    return conn


def row_to_dict(row):
    metadata = json.loads(row["metadata"])
    return {
        "id": row["id"],
        "title": metadata["title"],
        "abstract": metadata["abstract"],
        "authors": metadata["authors"].split(",") if metadata["authors"] else [],
        "year": metadata["year"],
        "keywords": metadata["mesh_terms"].split(";") if metadata["mesh_terms"] else [],
        "urls": "".join(metadata["urls"]).split(";") if metadata["urls"] else [],
    }

# --- 3. CORS Configuration (Crucial for Frontend/Backend communication) ---
# When your React app runs on, say, http://localhost:3000 and your FastAPI 
# runs on http://localhost:8000, you need CORS to allow the browser to make requests.
origins = [
    # Replace this with the actual URL of your React frontend in production.
    "http://localhost:8080",  
    "http://127.0.0.1:8080",
    "https://know-biol-s-space-app.vercel.app",
    # You can add more origins here if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/publications/count")
def get_publications_count():
    """Return the total number of publications in the database."""
    conn = get_db_connection()
    try:
        # Execute the SQL COUNT function, which returns a single row with the total count.
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings_queue")
        total_count = cursor.fetchone()[0]
        return {"count": total_count}
    except Exception as e:
        print(f"Database error during count: {e}")
        # Return 0 or handle error gracefully if needed
        return {"count": 0}
    finally:
        conn.close()
        
        
@app.get("/publications")
def get_publications(skip: int = 0, limit: int = 20):
    """Return a paginated list of publications"""
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT * FROM embeddings_queue LIMIT ? OFFSET ?", (limit, skip)
    ).fetchall()
    conn.close()
    return [row_to_dict(row) for row in rows]


@app.get("/publications/{pub_id}")
def get_publication(pub_id: str):
    """Return details of a single publication"""
    conn = get_db_connection()
    row = conn.execute(
        "SELECT * FROM embeddings_queue WHERE id = ?", (pub_id,)
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Publication not found")
    return row_to_dict(row)

uri = "neo4j+s://6e7e44a1.databases.neo4j.io"
driver = GraphDatabase.driver(uri, auth=("neo4j", NEO4J_KEY))

@app.get("/search")
def search_publications(query: str = Query(..., min_length=2)):
    """Search embeddings_queue by title, abstract, author, or keywords"""
    q = f"%{query.lower()}%"
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT * FROM embeddings_queue
        WHERE LOWER(title) LIKE ?
           OR LOWER(abstract) LIKE ?
           OR LOWER(authors) LIKE ?
           OR LOWER(keywords) LIKE ?
        """,
        (q, q, q, q),
    ).fetchall()
    conn.close()
    return [row_to_dict(row) for row in rows]


@app.get("/graph")
def get_graph(limit: int = 50):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Publication)
            OPTIONAL MATCH (p)-[:HAS_CONCEPT]->(m:Concepts)
            OPTIONAL MATCH (p)-[:HAS_CHEMICAL]->(c:Chemical)
            OPTIONAL MATCH (p)<-[:AUTHORED]-(a:Author)
            WITH p, collect(DISTINCT m) AS mesh, collect(DISTINCT c) AS chems, collect(DISTINCT a) AS authors
            RETURN 
              p {
                .id, .title, .pubYear, .doi, .abstract,
                mesh: [x IN mesh | {id: x.term, name: x.term, group: 3}],
                chems: [x IN chems | {id: x.name, name: x.name, group: 4}],
                authors: [x IN authors | {id: x.name, name: x.name}]
              }
            LIMIT $limit
            """,
            {"limit": limit}
        )

        nodes = {}
        links = []

        for record in result:
            pub = record["p"]

            # store publication metadata only in side panel
            pub_id = f"pub:{pub['id']}"

            for m in pub["mesh"]:
                mid = f"mesh:{m['id']}"
                nodes[mid] = m
                links.append({"source": mid, "target": pub_id})

            for c in pub["chems"]:
                cid = f"chem:{c['id']}"
                nodes[cid] = c
                links.append({"source": cid, "target": pub_id})

        return {
            "nodes": list(nodes.values()),
            "links": links,
        }
 

@app.post("/summarize")
def summarize_publication(title: str, abstract: str):
    summary_prompt = f"""The below is an abstract of a publication, I need you to not only summarize it for a reader.
    \\n #Title: \n{title} \\n #Abstract: \n{abstract}"""
    
    # Generate the summary through the predefined llm
    try:
        llm = init_chat_model("google_genai:gemini-2.5-flash", temperature=0.7)
        result= llm.invoke(summary_prompt)
        return result.content
    except Exception as e:
        raise e
        
# --- 5. Application Run Command ---
if __name__ == "__main__":
    # This command starts the Uvicorn server, which hosts the FastAPI application.
    # The application will be accessible at http://127.0.0.1:8000
    # The interactive documentation (Swagger UI) is at http://127.0.0.1:8000/docs
    uvicorn.run(app, host="127.0.0.1", port=8000)
