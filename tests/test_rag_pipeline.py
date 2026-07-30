import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.rag.vector_store import ComplaintVectorStore
from src.rag.retriever import ComplaintRetriever
from src.rag.generator import ComplaintGenerator
from src.rag.pipeline import ComplaintRAGPipeline
from src.rag.evaluate import evaluate


def test_rag_pipeline_end_to_end():
    """
    Minimal end-to-end test to verify that:
    - Vector store loads
    - Retrieval returns results
    - RAG pipeline returns a structured answer
    - Evaluation runs without error
    """

    # --- Vector Store ---
    vector_store = ComplaintVectorStore(
        parquet_path="data/complaint_embeddings.parquet"
    )
    vector_store.load_or_build()

    assert vector_store.index is not None
    assert vector_store.index.ntotal > 0

    # --- Retriever ---
    retriever = ComplaintRetriever(vector_store=vector_store)

    results = retriever.retrieve(
        "What are the common complaints about credit cards?",
        k=3
    )

    assert isinstance(results, list)
    assert len(results) > 0
    assert "text" in results[0]
    assert "metadata" in results[0]
    assert "score" in results[0]

    # --- Generator + Pipeline ---
    generator = ComplaintGenerator()
    pipeline = ComplaintRAGPipeline(
        retriever=retriever,
        generator=generator
    )

    response = pipeline.answer(
        "Do customers complain about unauthorized transactions?"
    )

    assert "question" in response
    assert "answer" in response
    assert "sources" in response
    assert len(response["sources"]) > 0

    # --- Evaluation ---
    eval_df = evaluate(pipeline)

    assert isinstance(eval_df, pd.DataFrame)
    assert not eval_df.empty
