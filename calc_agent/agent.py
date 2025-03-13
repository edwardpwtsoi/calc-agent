from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain.tools import tool
import ast
import re

class CalcTool(BaseTool):
    name = "calculator"
    description = "Useful for executing mathematical calculations from natural language"
    
    def _run(self, query: str, args: Optional[List[Any]] = None) -> float:
        """Execute the calculation based on the query and arguments"""
        try:
            # First get the code from the query
            generated_code = self._generate_code(query)
            
            # Execute the code with provided arguments
            result = self._execute_code(generated_code, args or [])
            return result
        except Exception as e:
            return f"Error executing calculation: {str(e)}"

    def _generate_code(self, query: str) -> str:
        """Generate Python code from natural language query"""
        # This will be implemented by the LLM
        raise NotImplementedError
        
    def _execute_code(self, code: str, args: List[Any]) -> float:
        """Safely execute the generated code with arguments"""
        # Basic security check - only allow mathematical operations
        if not self._is_safe_code(code):
            raise ValueError("Invalid code - only mathematical operations allowed")
            
        namespace = {}
        try:
            exec(code, namespace)
            # Find the function name in the code
            func_match = re.search(r"def\s+(\w+)\s*\(", code)
            if not func_match:
                raise ValueError("No function definition found in generated code")
                
            func_name = func_match.group(1)
            func = namespace[func_name]
            return func(*args)
        except Exception as e:
            raise RuntimeError(f"Error executing code: {str(e)}")

    def _is_safe_code(self, code: str) -> bool:
        """Check if the code contains only safe mathematical operations"""
        try:
            tree = ast.parse(code)
            # Here you would implement security checks
            # For now, we'll just verify it's valid Python
            return True
        except SyntaxError:
            return False 