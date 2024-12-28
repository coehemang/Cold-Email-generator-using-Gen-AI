import pandas as pd
import chromadb
import uuid
import requests
from io import StringIO

shared_link = "https://www.dropbox.com/scl/fi/68pp185mpnbzple629h5c/my_portfolio.csv?rlkey=5mgk30vpmo0gc4as0njzs8up5&st=3gtuzx2y&dl=1"

def fetch_csv_from_dropbox_url(url):
        response = requests.get(url)
        response.raise_for_status()  
        csv_data = pd.read_csv(StringIO(response.text))
        return csv_data  



class Portfolio:
    def __init__(self):
        self.data = fetch_csv_from_dropbox_url(shared_link)
        self.chroma_client = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(documents=row["Techstack"],
                                    metadatas={"links": row["Links"]},
                                    ids=[str(uuid.uuid4())])

    def query_links(self, skills):
        return self.collection.query(query_texts=skills, n_results=2).get('metadatas', [])
