from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from langchain_openai import AzureChatOpenAI
from langchain_anthropic import BedrockAnthropicModel


class LLMProvider(ABC):
    @abstractmethod
    def generate_code(self, query: str) -> str:
        """Generate Python code from natural language query"""
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, llm: Any):
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
        self.prompt = """
        You are a Python code generator. Convert the following natural language query into a Python function.
        The function should only contain mathematical operations.
        
        Query: {query}
        
        Return only the Python function code, no explanations.
        """
        
    def generate_code(self, query: str) -> str:
        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": self.prompt.format(query=query)
            }]
        )
        return response.content[0].text

class AzureOpenAIProvider(LLMProvider):
    def __init__(self, llm: AzureChatOpenAI):
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

class BedrockAnthropicProvider(LLMProvider):
    def __init__(self, llm: BedrockAnthropicModel):
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