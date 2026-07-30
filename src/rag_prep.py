import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import pickle
import logging
import sys

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss

# -----------------------------
# Configure logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("embedding_processor.log", mode="w")
    ]
)
logger = logging.getLogger(__name__)


# -----------------------------
# Embedding Processor Class
# -----------------------------
class EmbeddingProcessor:
    """
    Handles sampling, text chunking, embedding, and FAISS vector store indexing.

    Workflow:
    1. Load cleaned CSV dataset.
    2. Stratified sampling by product category.
    3. Split complaint narratives into overlapping chunks.
    4. Generate sentence embeddings for each chunk.
    5. Build and save FAISS vector index with associated metadata.
    """

    def __init__(
        self,
        cleaned_csv_path: str,
        vector_store_dir: str,
        sample_size: int = 15000,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        random_state: int = 42
    ):
        """
        Initialize the EmbeddingProcessor.

        Args:
            cleaned_csv_path (str): Path to the cleaned CSV dataset.
            vector_store_dir (str): Directory to save FAISS index and metadata.
            sample_size (int): Total number of rows to sample from dataset.
            chunk_size (int): Maximum characters per text chunk.
            chunk_overlap (int): Overlap in characters between consecutive chunks.
            embedding_model_name (str): SentenceTransformer model to use.
            random_state (int): Random seed for reproducible sampling.
        """
        self.cleaned_csv_path = Path(cleaned_csv_path)
        self.vector_store_dir = Path(vector_store_dir)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)

        self.sample_size = sample_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name
        self.random_state = random_state

        self.df = None
        self.sample_df = None
        self.chunks = []
        self.metadata = []

        try:
            self.model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.exception(f"Failed to load embedding model: {e}")
            raise

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        logger.info("EmbeddingProcessor initialized")

    # -----------------------------
    # Load cleaned dataset
    # -----------------------------
    def load_data(self):
        """
        Load the cleaned CSV dataset into memory.

        Returns:
            pd.DataFrame: Loaded dataset.

        Raises:
            FileNotFoundError: If CSV file does not exist.
            Exception: For other I/O or parsing errors.
        """
        try:
            self.df = pd.read_csv(self.cleaned_csv_path)
            logger.info(f"Loaded cleaned dataset with {len(self.df):,} rows")
        except FileNotFoundError:
            logger.exception(f"CSV file not found: {self.cleaned_csv_path}")
            raise
        except Exception as e:
            logger.exception(f"Error loading CSV: {e}")
            raise
        return self.df

    # -----------------------------
    # Stratified sampling
    # -----------------------------
    def stratified_sample(self, product_col="Product"):
        """
        Perform stratified sampling across product categories.

        Ensures approximately equal representation of each product in the sample.

        Args:
            product_col (str): Column name for product categories.

        Returns:
            pd.DataFrame: Sampled dataset.
        """
        try:
            num_categories = self.df[product_col].nunique()
            per_category = int(self.sample_size / num_categories)

            self.sample_df = (
                self.df.groupby(product_col, group_keys=False)
                .apply(lambda x: x.sample(min(len(x), per_category), random_state=self.random_state))
                .reset_index(drop=True)
            )
            logger.info(f"Stratified sample contains {len(self.sample_df):,} rows across {num_categories} categories")
        except Exception as e:
            logger.exception(f"Error during stratified sampling: {e}")
            raise
        return self.sample_df

    # -----------------------------
    # Chunk text and store metadata
    # -----------------------------
    def chunk_texts(self, text_col="Consumer complaint narrative", product_col="Product"):
        """
        Split complaint narratives into smaller overlapping chunks.

        Also stores metadata including complaint ID, product, and chunk index.

        Args:
            text_col (str): Column containing complaint text.
            product_col (str): Column containing product category.

        Returns:
            tuple: (list of chunk texts, list of metadata dicts)
        """
        self.chunks = []
        self.metadata = []

        try:
            for idx, row in tqdm(self.sample_df.iterrows(), total=len(self.sample_df), desc="Chunking Text"):
                text = str(row[text_col])
                chunk_texts = self.text_splitter.split_text(text)

                for i, chunk in enumerate(chunk_texts):
                    self.chunks.append(chunk)
                    self.metadata.append({
                        "complaint_id": row["Complaint ID"],
                        "product": row[product_col],
                        "chunk_index": i,
                        "total_chunks": len(chunk_texts)
                    })

            logger.info(f"Total chunks created: {len(self.chunks):,}")
        except Exception as e:
            logger.exception(f"Error during text chunking: {e}")
            raise

        return self.chunks, self.metadata

    # -----------------------------
    # Embed all chunks
    # -----------------------------
    def embed_chunks(self, batch_size=512):
        """
        Generate embeddings for all text chunks using the sentence transformer model.

        Args:
            batch_size (int): Number of chunks to embed in each batch.

        Returns:
            np.ndarray: Array of embeddings.
        """
        all_embeddings = []

        try:
            for i in tqdm(range(0, len(self.chunks), batch_size), desc="Embedding Chunks"):
                batch = self.chunks[i:i+batch_size]
                embeddings = self.model.encode(batch, show_progress_bar=False)
                all_embeddings.append(embeddings)

            self.embeddings = np.vstack(all_embeddings)
            logger.info(f"Generated embeddings with shape: {self.embeddings.shape}")
        except Exception as e:
            logger.exception(f"Error during embedding: {e}")
            raise

        return self.embeddings

    # -----------------------------
    # Build FAISS vector store
    # -----------------------------
    def build_faiss_index(self):
        """
        Create a FAISS vector index from embeddings and save it along with metadata.

        Saves:
        - FAISS index file: faiss_index.idx
        - Metadata pickle: metadata.pkl
        """
        try:
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.embeddings)
            logger.info(f"FAISS index created with {self.index.ntotal} vectors")

            # Save index and metadata
            faiss.write_index(self.index, str(self.vector_store_dir / "faiss_index.idx"))
            with open(self.vector_store_dir / "metadata.pkl", "wb") as f:
                pickle.dump(self.metadata, f)
            logger.info(f"FAISS index and metadata saved in {self.vector_store_dir}")
        except Exception as e:
            logger.exception(f"Error building FAISS index: {e}")
            raise

    # -----------------------------
    # Run the full pipeline
    # -----------------------------
    def run_pipeline(self):
        """
        Execute the complete Task 2 pipeline:
        1. Load cleaned CSV
        2. Stratified sampling
        3. Chunk texts
        4. Generate embeddings
        5. Build FAISS index and save metadata
        """
        try:
            self.load_data()
            self.stratified_sample()
            self.chunk_texts()
            self.embed_chunks()
            self.build_faiss_index()
            logger.info("Task 2 pipeline completed successfully")
        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            raise
