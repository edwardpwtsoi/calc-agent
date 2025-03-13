from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from langchain_core.language_models import BaseLLM

class LLMProvider(ABC):
    @abstractmethod
    def generate_code(self, query: str) -> str:
        """Generate Python code from natural language query"""
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.prompt = """
        You are a Python code generator. Convert the following natural language query into a Python function.
        The function should only contain mathematical operations.
        
        Query: {query}
        
        Return only the Python function code, no explanations.
        """
        
    def generate_code(self, query: str) -> str:
        response = self.llm.invoke(self.prompt.format(query=query))
        return response 

class AnthropicProvider(LLMProvider):
    def __init__(self, anthropic_client):
        self.client = anthropic_client
        
    def generate_code(self, query: str) -> str:
        # Implement Anthropic-specific code generation
        pass 