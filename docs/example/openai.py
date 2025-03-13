from langchain_openai import ChatOpenAI
from calc_agent.calculator import CalculatorAgent
from calc_agent.llm_provider import OpenAIProvider


def main():
    # Initialize with OpenAI
    llm = ChatOpenAI()
    provider = OpenAIProvider(llm)
    
    # Create calculator agent
    calculator = CalculatorAgent(provider)
    
    # Example usage
    query = "Create a function to calculate the area of a triangle given base and height"
    result = calculator.calculate(query, [5, 3])
    print(f"Area of triangle: {result}")
    
    # Another example
    query = "Write a function to find the average of three numbers"
    result = calculator.calculate(query, [10, 20, 30])
    print(f"Average: {result}")


if __name__ == "__main__":
    main()
