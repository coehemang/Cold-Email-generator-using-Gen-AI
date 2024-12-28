import pandas as pd
from sentence_transformers import SentenceTransformer
import requests
from io import StringIO
from pinecone import Pinecone, ServerlessSpec
import uuid

# Dropbox shared link for the portfolio data
shared_link = "https://www.dropbox.com/scl/fi/68pp185mpnbzple629h5c/my_portfolio.csv?rlkey=5mgk30vpmo0gc4as0njzs8up5&st=3gtuzx2y&dl=1"


def fetch_csv_from_dropbox_url(url):
        response = requests.get(url)
        response.raise_for_status()
        csv_data = pd.read_csv(StringIO(response.text))
        return csv_data


class Portfolio:
    def __init__(self):
        self.data = fetch_csv_from_dropbox_url(shared_link)
        self.pinecone_client = Pinecone(api_key="pcsk_2nQXbL_AmyBNAPz7zz9DwXe3mDr65yFU1rF6EJf9f1vQbZir4SzBLk9DTQBVq5CVDUqRP2")
        self.index_name = "portfolioindex" 

        if self.index_name not in [index["name"] for index in self.pinecone_client.list_indexes().indexes]:
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=384,  
                metric="cosine",  
                spec=ServerlessSpec(cloud="aws", region="us-east-1")  
            )

        self.index = self.pinecone_client.Index(self.index_name)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def load_portfolio(self):
        # Get index stats
        stats = self.index.describe_index_stats()

        # Check if the index is empty
        if stats["namespaces"].get("", {}).get("vector_count", 0) == 0:
            vectors = []
            for _, row in self.data.iterrows():
                # Compute embeddings for Techstack
                embedding = self.model.encode(row["Techstack"]).tolist()
                vectors.append({
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": {"links": row["Links"]},
                })

            # Upsert vectors into the Pinecone index
            self.index.upsert(vectors)

    def query_links(self, skills):
        query_embedding = self.model.encode(skills).tolist()

        # Query the Pinecone index
        results = self.index.query(vector=query_embedding, top_k=2, include_metadata=True)

        # Extract metadata containing links
        return [match["metadata"]["links"] for match in results["matches"]]
