from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from config.settings import LLMFactory


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class RAGGraphBuilder:
    """Assembles LangGraph engine state machine workflow topologies."""

    def __init__(self, tools_list: list):
        self.tools_list = tools_list
        self.tool_node = ToolNode(tools_list)
        self.orchestrator = LLMFactory.get_orchestrator_llm().bind_tools(tools_list)

    def _call_model(self, state: AgentState):
        """Standard wrapper node processing engine generation requests."""
        response = self.orchestrator.invoke(state["messages"])
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> str:
        """Conditional routing parser checking tool calls status."""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def build(self):
        """Assembles state transitions into an executable machine pipeline."""
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self.tool_node)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", self._should_continue, {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()
