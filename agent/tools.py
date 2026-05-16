from langchain_core.tools import tool
from langchain_experimental.agents import create_spark_dataframe_agent
from config.settings import logger, LLMFactory


def create_agent_tools(retriever, sales_df):
    """Factory creating operational capability sets with isolated worker environments."""

    spark_internal_llm = LLMFactory.get_spark_internal_llm()
    spark_agent_executor = create_spark_dataframe_agent(
        llm=spark_internal_llm,
        df=sales_df,
        verbose=False,
        allow_dangerous_code=True,
    )

    @tool
    def query_knowledge_base(query: str) -> str:
        """Searches corporate policy and unstructured knowledge documentation."""
        try:
            docs = retriever.invoke(query)
            return "\n\n".join([str(d.page_content) for d in docs])
        except Exception as e:
            logger.error(f"Knowledge base query failed: {e}")
            return "Error querying knowledge base."

    @tool
    def query_sales_records(query: str) -> str:
        """Queries transactional sales databases containing Revenue, Employees, and Departments."""
        try:
            response = spark_agent_executor.invoke({"input": query})
            if isinstance(response, dict) and "output" in response:
                return str(response["output"])
            return str(response)
        except Exception as e:
            logger.error(f"Sales records query failed: {e}")
            return "Error querying sales records."

    return [query_knowledge_base, query_sales_records]
