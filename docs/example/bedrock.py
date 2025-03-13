import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from calc_agent.calculator import CalculatorAgent
from calc_agent.llm_provider import BedrockAnthropicProvider


def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize with Amazon Bedrock using environment variables
    llm = ChatBedrock(
        model_id=os.getenv('BEDROCK_MODEL_ID'),
        region_name=os.getenv('BEDROCK_REGION_NAME'),
        model_kwargs={
            "max_tokens": 1000,
            "temperature": 0.5,
        }
    )
    provider = BedrockAnthropicProvider(llm)
    
    # Create calculator agent
    calculator = CalculatorAgent(provider)
    
    # Example usage
    query = "Calculate the area of a triangle given base 5 and height 3"
    print(f"Query: {query}")
    result = calculator.calculate(query)
    print(f"Area of triangle: {result}")
    
    # Another example
    query = "Calculate the average of 10, 20, and 30"
    print(f"Query: {query}")
    result = calculator.calculate(query)
    print(f"Average: {result}")


if __name__ == "__main__":
    main()
