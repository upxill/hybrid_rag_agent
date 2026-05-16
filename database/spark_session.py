from pyspark.sql import SparkSession
from config.settings import logger


class SparkManager:
    """Handles Spark cluster context lifecycle."""

    def __init__(self):
        self.spark = (
            SparkSession.builder.appName("SparkRAGAgent")
            .config("spark.driver.memory", "4g")
            .getOrCreate()
        )
        logger.info("Spark session initialized successfully.")

    def create_sales_dataframe(self):
        """Creates sample structured mock sales metrics schema."""
        sales_data = [
            ("Alice", "Electronics", 1200, "2026-01-15"),
            ("Bob", "Software", 450, "2026-02-11"),
            ("Charlie", "Electronics", 2900, "2026-03-01"),
            ("Alice", "Software", 150, "2026-04-10"),
        ]
        schema = ["Employee", "Department", "Revenue", "TransactionDate"]
        return self.spark.createDataFrame(sales_data, schema)
