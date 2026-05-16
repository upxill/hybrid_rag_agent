import logging
import os
import sys
from langchain_openai import ChatOpenAI

# Centralized logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SparkRAGAgent")

# Directory pathways
INPUT_DATA_DIR = "data/raw_inputs/"
SPARK_CHECKPOINT_DIR = "data/spark_checkpoints/"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "corporate_knowledge"


def setup_environment():
    """Aligns worker pathways and validates key presence."""
    venv_python = sys.executable
    os.environ["PYSPARK_PYTHON"] = venv_python
    os.environ["PYSPARK_DRIVER_PYTHON"] = venv_python

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # Force auto-creation of missing data and checkpoint folders
    os.makedirs(INPUT_DATA_DIR, exist_ok=True)
    os.makedirs(SPARK_CHECKPOINT_DIR, exist_ok=True)


class LLMFactory:
    """Manages creation of discrete model instances."""

    @staticmethod
    def get_orchestrator_llm():
        return ChatOpenAI(model="gpt-4o", temperature=0)

    @staticmethod
    def get_spark_internal_llm():
        return ChatOpenAI(model="gpt-4o", temperature=0)
