import uuid
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import (
    logger,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    INPUT_DATA_DIR,
    SPARK_CHECKPOINT_DIR,
)


# --- Distributed Spark Chunking Function ---
def split_text_chunks(content: str) -> list:
    """Splits document content natively across distributed Spark worker nodes."""
    if not content:
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    return [doc.page_content for doc in text_splitter.create_documents([content])]


split_chunks_udf = F.udf(split_text_chunks, ArrayType(StringType()))


# --- Spark Streaming Batch Target Writing Sink ---
def write_to_chroma_batch(batch_df, batch_id):
    """Processes micro-batches from the Spark Stream and pushes to Chroma."""
    rows = batch_df.collect()
    if not rows:
        return

    logger.info(
        f"Processing batch {batch_id} containing {len(rows)} new text fragments."
    )

    chunks = []
    metadatas = []
    ids = []

    for row in rows:
        chunks.append(row["chunk"])
        metadatas.append({"source": row["source_file"]})
        ids.append(str(uuid.uuid4()))

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
    vector_store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
    vector_store.persist()
    logger.info(f"Batch {batch_id} committed to Chroma database store instance.")


# --- Database Ingestion Manager Class ---
class SparkVectorStoreManager:
    """Uses Spark Structured Streaming to incrementally sync files into Chroma."""

    def __init__(self, spark_session: SparkSession):
        self.spark = spark_session
        self._ensure_database_exists()

    def _ensure_database_exists(self):
        """Initializes and persists Chroma index on disk if not already established."""
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=OpenAIEmbeddings(),
            persist_directory=CHROMA_PERSIST_DIR,
        )
        vector_store.persist()
        logger.info("ChromaDB layer checked and confirmed online.")

    def sync_new_files(self):
        """Runs incremental AvailableNow stream scanning directory for new documents."""
        logger.info("Scanning for new file systems data via Spark Ingestion...")
        file_schema = StructType([StructField("value", StringType(), True)])

        raw_stream_df = (
            self.spark.readStream.format("text")
            .schema(file_schema)
            .load(INPUT_DATA_DIR)
            .select(
                F.col("value").alias("raw_content"),
                F.input_file_name().alias("source_file"),
            )
        )

        chunked_stream_df = (
            raw_stream_df.withColumn(
                "chunks_array", split_chunks_udf(F.col("raw_content"))
            )
            .withColumn("chunk", F.explode(F.col("chunks_array")))
            .select("chunk", "source_file")
        )

        query = (
            chunked_stream_df.writeStream.foreachBatch(write_to_chroma_batch)
            .option("checkpointLocation", SPARK_CHECKPOINT_DIR)
            .trigger(availableNow=True)
            .start()
        )
        query.awaitTermination()
        logger.info("Incremental file synchronization complete.")

    def get_retriever(self):
        """Yields looking client for LangGraph agent processing queries."""
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=OpenAIEmbeddings(),
            persist_directory=CHROMA_PERSIST_DIR,
        )
        return vector_store.as_retriever(search_kwargs={"k": 1})
