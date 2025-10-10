"""Upload dataset to Langfuse with metadata support."""
from pathlib import Path
from shared.clients import get_langfuse_client
from shared.datasets import upload_langfuse_dataset

SCRIPT_DIR = Path(__file__).parent
DATASET_FILE = SCRIPT_DIR / "dataset.jsonl"
DATASET_NAME = "dataset_docgen"

langfuse = get_langfuse_client()

count = upload_langfuse_dataset(
    DATASET_FILE,
    DATASET_NAME,
    "Dataset for pairwise comparison of documentation prompts",
    langfuse,
    metadata_override={
        "source": "5-langfuse"
    }
)

print("="*60)
print(f"Dataset '{DATASET_NAME}' updated with {count} examples")
print()
print("Access Langfuse UI at: http://localhost:3000")
print("Navigate to: Datasets > dataset_docgen")
