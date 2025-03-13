from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Optional
from langchain_openai import AzureChatOpenAI
from langchain_aws import BedrockLLM
from langchain_community.llms.anthropic import Anthropic
import json


class LLMProvider(ABC):
    @abstractmethod
    def generate_code_and_args(self, query: str) -> Dict[str, Any]:
        """Generate Python code and arguments from natural language query"""
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
        return str(response)

class AnthropicProvider(LLMProvider):
    def __init__(self, anthropic_client: Anthropic):
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
        return str(response)

class BedrockAnthropicProvider(LLMProvider):
    def __init__(self, llm: BedrockLLM):
        self.llm = llm
        self.prompt = """
        You are a Python code generator. Convert the following natural language query into a Python function and extract arguments.
        The function should only contain mathematical operations.
        
        Query: {query}
        
        Return the Python function code and arguments as a JSON object.
        
        Format:
        {{
            "code": ${{python code}},
            "arguments": ${{list of input values}}
        }}
        Respond with only the JSON object, no additional text or explanations."""
        
    def generate_code_and_args(self, query: str) -> Dict[str, Any]:
        print(f"Generating code and arguments for query: {query}")
        response = self.llm.invoke(self.prompt.format(query=query))
        print(f"Response: {response.content}")
        # Extract JSON from the response
        try:
            # Find JSON object in the response
            response_obj = json.loads(response.content)
            return response_obj
        except Exception as e:
            print(f"Failed to parse response: {response.content}")
            raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}")
