import pytest
from pathlib import Path
from src.EDA import EDAProcessor


@pytest.fixture
def processor(tmp_path):
    """Create a lightweight EDAProcessor instance for testing."""
    return EDAProcessor(
        parquet_path="dummy.parquet",
        output_csv_path=tmp_path / "out.csv",
        figures_dir=tmp_path / "figures"
    )


def test_clean_and_normalize_text(processor):
    raw_text = """
    I am writing to file a complaint.
    Please visit http://example.com
    My credit card was charged $500 on 01-01-2023.
    """

    cleaned = processor.clean_text_noise(raw_text)
    normalized = processor.normalize_text(cleaned)

    # Assertions
    assert isinstance(normalized, str)
    assert "http" not in normalized
    assert "writing" not in normalized  # boilerplate removed
    assert "credit" in normalized
    assert "card" in normalized
    assert len(normalized.split()) > 0
