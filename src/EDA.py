import pyarrow.parquet as pq
import pandas as pd
import re
import logging
from collections import Counter
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt
import seaborn as sns

# NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EDAProcessor:
    """
    End-to-end EDA and preprocessing pipeline for large-scale consumer complaint data.

    This class:
    - Streams large Parquet files in batches
    - Performs exploratory data analysis (EDA)
    - Cleans and normalizes complaint narratives
    - Filters by allowed financial product categories
    - Writes cleaned outputs incrementally to CSV
    - Generates and saves EDA visualizations
    """

    def __init__(self, parquet_path, output_csv_path, figures_dir, batch_size=50000):
        """
        Initialize the EDAProcessor.

        Args:
            parquet_path (str or Path): Path to the input Parquet dataset
            output_csv_path (str or Path): Destination CSV path for cleaned data
            figures_dir (str or Path): Directory where EDA plots will be saved
            batch_size (int): Number of rows to process per batch
        """
        self.parquet_path = parquet_path
        self.output_csv_path = output_csv_path
        self.batch_size = batch_size

        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.allowed_keywords = [
            "credit card",
            "personal loan",
            "savings account",
            "money transfer"
        ]

        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        self.total_rows = 0
        self.product_distribution = Counter()
        self.with_narrative = 0
        self.without_narrative = 0
        self.raw_word_lengths = []
        self.word_lengths = []

        logger.info("EDAProcessor initialized")

    # -----------------------------
    # Load batches
    # -----------------------------
    def load_batches(self):
        """
        Lazily load the Parquet dataset in batches.

        Yields:
            pd.DataFrame: A batch of rows converted to a pandas DataFrame
        """
        parquet = pq.ParquetFile(self.parquet_path)
        for batch in parquet.iter_batches(batch_size=self.batch_size):
            yield batch.to_pandas()

    # -----------------------------
    # EDA counters
    # -----------------------------
    def eda_step(self, df: pd.DataFrame):
        """
        Update high-level EDA statistics for a batch.

        Tracks:
        - Total rows processed
        - Product distribution
        - Presence or absence of complaint narratives

        Args:
            df (pd.DataFrame): Raw batch dataframe
        """
        self.total_rows += len(df)
        self.product_distribution.update(df["Product"].dropna().values)
        narrative = df["Consumer complaint narrative"]
        has_text = narrative.notna() & (narrative.str.strip() != "")
        self.with_narrative += has_text.sum()
        self.without_narrative += (~has_text).sum()

    # -----------------------------
    # Clean + normalize
    # -----------------------------
    def clean_text_noise(self, text):
        """
        Remove noise and boilerplate from raw complaint text.

        This includes:
        - URLs
        - Phone numbers
        - Account-like identifiers
        - HTML artifacts
        - Known boilerplate phrases
        - Punctuation

        Args:
            text (str): Raw complaint narrative

        Returns:
            str: Cleaned lowercase text
        """
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\b\d{3}[-.\s]?\d{4}\b', '', text)
        text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '', text)
        text = re.sub(r'\b[A-Z]{2}\d{10,}\b', '', text, flags=re.I)
        text = re.sub(r'<.*?>', '', text)
        boilerplate = [
            r"i am writing to file a complaint",
            r"this complaint is regarding",
            r"consumer complaint narrative"
        ]
        for bp in boilerplate:
            text = re.sub(bp, '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def normalize_text(self, text):
        """
        Normalize cleaned text via tokenization, stopword removal, and lemmatization.

        Steps:
        - Tokenize
        - Remove stopwords and short tokens
        - Lemmatize verbs and nouns

        Args:
            text (str): Cleaned text

        Returns:
            str: Normalized text
        """
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        lemmas = [self.lemmatizer.lemmatize(t, pos='v') for t in tokens]
        lemmas = [self.lemmatizer.lemmatize(t, pos='n') for t in lemmas]
        return ' '.join(lemmas)

    # -----------------------------
    # Filter and clean
    # -----------------------------
    def filter_and_clean(self, df: pd.DataFrame):
        """
        Filter complaints and apply text preprocessing.

        Steps:
        - Remove rows without narratives
        - Track raw narrative lengths
        - Filter to allowed product categories
        - Clean and normalize text
        - Track cleaned narrative lengths

        Args:
            df (pd.DataFrame): Raw batch dataframe

        Returns:
            pd.DataFrame: Filtered and cleaned dataframe
        """
        narrative = df["Consumer complaint narrative"].fillna("")
        has_text = narrative.str.strip() != ""
        df = df[has_text].copy()

        raw_lengths = narrative[has_text].str.split().str.len()
        self.raw_word_lengths.extend(raw_lengths.tolist())

        mask = df["Product"].str.lower().apply(
            lambda x: any(kw in x for kw in self.allowed_keywords)
        )
        df = df[mask].copy()
        if df.empty:
            return df

        df["processed_feedback"] = df["Consumer complaint narrative"].apply(self.clean_text_noise)
        df["normalized_feedback"] = df["processed_feedback"].apply(self.normalize_text)

        clean_lengths = df["normalized_feedback"].str.split().str.len()
        self.word_lengths.extend(clean_lengths.tolist())

        return df

    # -----------------------------
    # Save CSV
    # -----------------------------
    def save_csv(self, df: pd.DataFrame, write_header: bool):
        """
        Append cleaned data to the output CSV.

        Args:
            df (pd.DataFrame): Cleaned batch dataframe
            write_header (bool): Whether to write CSV header
        """
        df.to_csv(
            self.output_csv_path,
            mode='w' if write_header else 'a',
            header=write_header,
            index=False
        )

    # -----------------------------
    # Build EDA summary
    # -----------------------------
    def build_results(self):
        """
        Compile final EDA summary statistics.

        Returns:
            dict: Aggregated dataset statistics and narrative length metrics
        """
        raw_wl = pd.Series(self.raw_word_lengths)
        clean_wl = pd.Series(self.word_lengths)
        return {
            "total_rows": self.total_rows,
            "with_narrative": self.with_narrative,
            "without_narrative": self.without_narrative,
            "product_distribution": dict(self.product_distribution),
            "raw_narrative_length_stats": {
                "min": int(raw_wl.min()) if not raw_wl.empty else 0,
                "max": int(raw_wl.max()) if not raw_wl.empty else 0,
                "mean": round(float(raw_wl.mean()), 2) if not raw_wl.empty else 0,
                "median": int(raw_wl.median()) if not raw_wl.empty else 0
            },
            "clean_narrative_length_stats": {
                "min": int(clean_wl.min()) if not clean_wl.empty else 0,
                "max": int(clean_wl.max()) if not clean_wl.empty else 0,
                "mean": round(float(clean_wl.mean()), 2) if not clean_wl.empty else 0,
                "median": int(clean_wl.median()) if not clean_wl.empty else 0
            }
        }

    # -----------------------------
    # Separate plots
    # -----------------------------
    def plot_product_distribution(self):
        """
        Plot and save complaint counts by product category.
        """
        plt.figure(figsize=(8, 4))
        sns.barplot(
            x=list(self.product_distribution.keys()),
            y=list(self.product_distribution.values())
        )
        plt.title("Complaint Distribution by Product")
        plt.xticks(rotation=30)
        plt.tight_layout()
        path = self.figures_dir / "product_distribution.png"
        plt.savefig(path)
        plt.close()
        logger.info(f"Product distribution plot saved: {path}")

    def plot_narrative_availability(self):
        """
        Plot and save counts of complaints with and without narratives.
        """
        plt.figure(figsize=(6, 4))
        sns.barplot(
            x=["With Narrative", "Without Narrative"],
            y=[self.with_narrative, self.without_narrative]
        )
        plt.title("Narrative Availability")
        plt.tight_layout()
        path = self.figures_dir / "narrative_availability.png"
        plt.savefig(path)
        plt.close()
        logger.info(f"Narrative availability plot saved: {path}")

    def plot_raw_word_lengths(self):
        """
        Plot and save histogram of raw narrative word counts.
        """
        plt.figure(figsize=(8, 5))
        sns.histplot(pd.Series(self.raw_word_lengths), bins=50, color='skyblue')
        plt.title("Raw Narrative Word Counts")
        plt.xlabel("Word Count")
        plt.ylabel("Number of Complaints")
        plt.tight_layout()
        path = self.figures_dir / "raw_word_lengths.png"
        plt.savefig(path)
        plt.close()
        logger.info(f"Raw word length histogram saved: {path}")

    def plot_clean_word_lengths(self):
        """
        Plot and save histogram of cleaned narrative word counts.
        """
        plt.figure(figsize=(8, 5))
        sns.histplot(pd.Series(self.word_lengths), bins=50, color='orange')
        plt.title("Clean Narrative Word Counts")
        plt.xlabel("Word Count")
        plt.ylabel("Number of Complaints")
        plt.tight_layout()
        path = self.figures_dir / "clean_word_lengths.png"
        plt.savefig(path)
        plt.close()
        logger.info(f"Clean word length histogram saved: {path}")

    # -----------------------------
    # Run pipeline
    # -----------------------------
    def run(self):
        """
        Execute the full EDA and preprocessing pipeline.

        Returns:
            dict: Final EDA summary statistics
        """
        write_header = True
        for batch_num, df in enumerate(self.load_batches(), start=1):
            logger.info(f"Processing batch {batch_num} with {len(df):,} rows")
            self.eda_step(df)
            cleaned = self.filter_and_clean(df)
            if not cleaned.empty:
                self.save_csv(cleaned, write_header)
                write_header = False

        results = self.build_results()
        logger.info("Task 1 EDA + preprocessing complete")
        return results
