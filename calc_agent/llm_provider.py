from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Optional, Callable
from langchain_openai import AzureChatOpenAI
from langchain_aws import BedrockLLM
from langchain_community.llms.anthropic import Anthropic
import json
import time
from functools import wraps


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1,
    exponential_base: float = 2,
    max_delay: float = 10
) -> Callable:
    """
    Decorator that implements exponential backoff retry logic
    
    Args:
        max_retries: Maximum number of retries before giving up
        initial_delay: Initial delay between retries in seconds
        exponential_base: Base for exponential backoff
        max_delay: Maximum delay between retries in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for retry in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if retry == max_retries - 1:
                        logging.error(f"Failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    wait_time = min(delay * (exponential_base ** retry), max_delay)
                    logging.warning(f"Attempt {retry + 1} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)
            
            raise last_exception
        return wrapper
    return decorator


class LLMProvider(ABC):
    def __init__(self):
        self.base_prompt = """
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

    @abstractmethod
    def generate_code_and_args(self, query: str) -> Dict[str, Any]:
        """Generate Python code and arguments from natural language query"""
        pass

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Helper method to parse JSON response"""
        try:
            return json.loads(response)
        except Exception as e:
            logging.error(f"Failed to parse response: {response}")
            raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}")

class OpenAIProvider(LLMProvider):
    def __init__(self, llm: Any):
        super().__init__()
        self.llm = llm
        self.prompt = """
        You are a Python code generator. Convert the following natural language query into a Python function.
        The function should only contain mathematical operations.
        
        Query: {query}
        
        Return only the Python function code, no explanations.
        """
        
    @retry_with_exponential_backoff()
    def generate_code_and_args(self, query: str) -> Dict[str, Any]:
        response = self.llm.invoke(self.prompt.format(query=query))
        return self._parse_json_response(str(response))

class AnthropicProvider(LLMProvider):
    def __init__(self, anthropic_client: Anthropic):
        super().__init__()
        self.client = anthropic_client
        self.prompt = """
        You are a Python code generator. Convert the following natural language query into a Python function.
        The function should only contain mathematical operations.
        
        Query: {query}
        
        Return only the Python function code, no explanations.
        """
        
    @retry_with_exponential_backoff()
    def generate_code_and_args(self, query: str) -> Dict[str, Any]:
        response = self.client.invoke(self.base_prompt.format(query=query))
        return self._parse_json_response(str(response))

class AzureOpenAIProvider(LLMProvider):
    def __init__(self, llm: AzureChatOpenAI):
        super().__init__()
        self.llm = llm
        self.prompt = """
        You are a Python code generator. Convert the following natural language query into a Python function.
        The function should only contain mathematical operations.
        
        Query: {query}
        
        Return only the Python function code, no explanations.
        """
        
    @retry_with_exponential_backoff()
    def generate_code_and_args(self, query: str) -> Dict[str, Any]:
        response = self.llm.invoke(self.base_prompt.format(query=query))
        return self._parse_json_response(str(response))

class BedrockAnthropicProvider(LLMProvider):
    def __init__(self, llm: BedrockLLM):
        super().__init__()
        self.llm = llm
        
    @retry_with_exponential_backoff()
    def generate_code_and_args(self, query: str) -> Dict[str, Any]:
        """Generate code and arguments from the query using Bedrock"""
        logging.info(f"Generating code and arguments for query: {query}")
        
        # Get response from LLM
        response = self.llm.invoke(self.base_prompt.format(query=query))
        # Log token usage
        if hasattr(response, 'usage'):
            logging.info(f"Token usage - Input: {response.usage.input_tokens}, Output: {response.usage.output_tokens}, Total: {response.usage.total_tokens}")
        
        # Parse and return the response
        return self._parse_json_response(response.content)
