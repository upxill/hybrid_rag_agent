from config.settings import setup_environment, logger
from database.spark_session import SparkManager
from database.vector_store import SparkVectorStoreManager
from agent.tools import create_agent_tools
from agent.graph import RAGGraphBuilder
from langchain_core.messages import HumanMessage


def main():
    try:
        # 1. Initialize configuration and environmental layers
        setup_environment()

        # 2. Bootstrap computing databases/data layers
        spark_mgr = SparkManager()
        sales_df = spark_mgr.create_sales_dataframe()

        vector_mgr = SparkVectorStoreManager(spark_mgr.spark)
        # Scan and stream new inputs into Chroma index database
        vector_mgr.sync_new_files()
        retriever = vector_mgr.get_retriever()

        # 3. Assemble and initialize the LangGraph system
        tools = create_agent_tools(retriever, sales_df)
        graph_builder = RAGGraphBuilder(tools)
        app = graph_builder.build()

        logger.info("Integrated Spark RAG Agent Engine successfully online.")

        # --- Interactive Query Execution Test Examples ---
        test_query = "What were Alice's total sales revenues?"
        response = app.invoke({"messages": [HumanMessage(content=test_query)]})
        print(f"Agent Response:\n{response['messages'][-1].content}")

    except Exception as e:
        logger.critical(
            f"Process crashed during execution configuration setup phases: {e}",
            exc_info=True,
        )


if __name__ == "__main__":
    main()
