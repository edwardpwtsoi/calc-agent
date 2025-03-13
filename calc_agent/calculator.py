from typing import Any, Dict, List
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from .agent import CalcTool
from .llm_provider import LLMProvider


class CalculatorAgent:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.calc_tool = CalcTool()
        
    def create_graph(self) -> Any:
        workflow = StateGraph(Dict)
        
        # Define the nodes
        def parse_input(state: Dict):
            query = state["query"]
            
            # Generate code and arguments using the LLM provider
            result = self.llm_provider.generate_code_and_args(query)
            
            return {
                "code": result["code"],
                "args": result["arguments"]
            }
            
        def execute_calculation(state: Dict):
            code = state["code"]
            args = state["args"]
            
            result = self.calc_tool._execute_code(code, args)
            
            return {
                "result": result
            }
            
        # Add nodes to graph
        workflow.add_node("parse_input", parse_input)
        workflow.add_node("execute_calculation", execute_calculation)
        
        # Define edges
        workflow.add_edge("parse_input", "execute_calculation")
        
        # Set entry and exit points
        workflow.set_entry_point("parse_input")
        workflow.set_finish_point("execute_calculation")
        
        return workflow.compile()

    def calculate(self, query: str, args: List[Any] | None = None) -> Any:
        """Execute a calculation from natural language query"""
        graph = self.create_graph()
        
        result = graph.invoke({
            "query": query,
            "args": args or []
        })
        
        return result["result"]
