import pandas as pd
from sentence_transformers import SentenceTransformer
import requests
from io import StringIO
from pinecone import Pinecone, ServerlessSpec
import uuid

# Dropbox shared link for the portfolio data
shared_link = "https://www.dropbox.com/scl/fi/68pp185mpnbzple629h5c/my_portfolio.csv?rlkey=5mgk30vpmo0gc4as0njzs8up5&st=3gtuzx2y&dl=1"


def fetch_csv_from_dropbox_url(url):
    """Fetches CSV data from the provided Dropbox URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        csv_data = pd.read_csv(StringIO(response.text))
        return csv_data
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error downloading CSV: {e}")


class Portfolio:
    def __init__(self):
        """Initializes the Portfolio class."""
        # Load portfolio data
        self.data = fetch_csv_from_dropbox_url(shared_link)

        # Initialize Pinecone with new method
        self.pinecone_client = Pinecone(api_key="pcsk_2nQXbL_AmyBNAPz7zz9DwXe3mDr65yFU1rF6EJf9f1vQbZir4SzBLk9DTQBVq5CVDUqRP2")
        self.index_name = "portfolioindex"  # Ensure the name is lowercase and valid

        # Check if index exists; if not, create it
        if self.index_name not in [index["name"] for index in self.pinecone_client.list_indexes().indexes]:
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=384,  # Ensure this matches the dimensionality of your embeddings
                metric="cosine",  # Choose the appropriate metric
                spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Replace with your desired cloud and region
            )

        # Connect to the index
        self.index = self.pinecone_client.Index(self.index_name)

        # Load the SentenceTransformer model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def load_portfolio(self):
        """Loads and embeds portfolio data into the Pinecone index."""
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
        """Queries the Pinecone index based on skills and returns relevant links."""
        # Compute the embedding for the query
        query_embedding = self.model.encode(skills).tolist()

        # Query the Pinecone index
        results = self.index.query(vector=query_embedding, top_k=2, include_metadata=True)

        # Extract metadata containing links
        return [match["metadata"]["links"] for match in results["matches"]]
