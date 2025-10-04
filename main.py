# main.py

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import sqlite3
import json
from neo4j import GraphDatabase
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
    # You can add more origins here if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)



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
async def get_graph(limit: int = 50):
    session = driver.session()

    query = """
      MATCH (p:Publication)
      OPTIONAL MATCH (p)-[:AUTHORED]->(a:Author)
      OPTIONAL MATCH (p)-[:HAS_Concept]->(m:Concepts)
      OPTIONAL MATCH (p)-[:HAS_CHEMICAL]->(c:Chemical)
      RETURN p, collect(distinct a) as authors, 
             collect(distinct m) as concepts, 
             collect(distinct c) as chemicals
      LIMIT $limit
    """

    result = session.run(query, limit=limit)

    nodes = []
    links = []
    node_set = set()

    for record in result:
        pub = record["p"]
        pub_id = pub["id"]

        # publication node
        if pub_id not in node_set:
            nodes.append({
                "id": pub_id,
                "name": pub.get("title", "Untitled"),
                "group": 1
            })
            node_set.add(pub_id)

        # authors
        for a in record["authors"]:
            if a:
                if a["id"] not in node_set:
                    nodes.append({"id": a["id"], "name": a["name"], "group": 2})
                    node_set.add(a["id"])
                links.append({"source": pub_id, "target": a["id"], "value": 1})

        # mesh terms
        for m in record["concepts"]:
            if m:
                if m["id"] not in node_set:
                    nodes.append({"id": m["id"], "name": m["name"], "group": 3})
                    node_set.add(m["id"])
                links.append({"source": pub_id, "target": m["id"], "value": 2})

        # chemicals
        for c in record["chemicals"]:
            if c:
                if c["id"] not in node_set:
                    nodes.append({"id": c["id"], "name": c["name"], "group": 4})
                    node_set.add(c["id"])
                links.append({"source": pub_id, "target": c["id"], "value": 2})

    session.close()
    return {"nodes": nodes, "links": links}

# --- 5. Application Run Command ---
if __name__ == "__main__":
    # This command starts the Uvicorn server, which hosts the FastAPI application.
    # The application will be accessible at http://127.0.0.1:8000
    # The interactive documentation (Swagger UI) is at http://127.0.0.1:8000/docs
    uvicorn.run(app, host="127.0.0.1", port=8000)
