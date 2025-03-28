# Natural Language Calculator Agent

A flexible calculator agent that converts natural language queries into executable Python functions using LLMs (Language Model Models). Built with LangChain and LangGraph, this tool supports multiple LLM providers and safely executes mathematical calculations.

## Features

- 🔢 Convert natural language to mathematical functions
- 🔒 Secure code execution with safety checks
- 🔄 Flexible LLM provider support (OpenAI, Anthropic, etc.)
- 📊 Graph-based workflow using LangGraph
- 🛠️ Easy to extend and customize

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/calc-agent.git
cd calc-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```
# Copy the example.env file and update with your configuration
cp example.env .env
```

## Usage

### Basic Usage

```python
from langchain_aws import ChatBedrock
from calc_agent.calculator import CalculatorAgent
from calc_agent.llm_provider import BedrockAnthropicProvider

# Initialize with Amazon Bedrock
llm = ChatBedrock(
    model_id="anthropic.claude-v2",  # or your chosen model
    region_name="us-east-1",         # your AWS region
    model_kwargs={
        "max_tokens": 1000,
        "temperature": 0.5,
    }
)
provider = BedrockAnthropicProvider(llm)
calculator = CalculatorAgent(provider)

# Calculate area of a triangle
result = calculator.calculate(
    "Calculate the area of a triangle given base 5 and height 3"
)
print(f"Area of triangle: {result}")  # Output: Area of triangle: 7.5

# Calculate average of numbers
result = calculator.calculate(
    "Calculate the average of 10, 20, and 30"
)
print(f"Average: {result}")  # Output: Average: 20.0
```

### Using Different LLM Providers

The agent supports multiple LLM providers. Here's how to use Anthropic:

```python
from calc_agent.llm_provider import AnthropicProvider
import anthropic

# Initialize Anthropic client
client = anthropic.Client(api_key="your_api_key")
provider = AnthropicProvider(client)
calculator = CalculatorAgent(provider)
```

## Project Structure

```
calc_agent/
├── agent.py        # Core calculation tool implementation
├── calculator.py   # Main calculator agent with graph workflow
├── llm_provider.py # LLM provider interfaces
docs/
└── example/
    └── openai.py   # OpenAI usage examples
```

## Safety and Limitations

- The agent only executes mathematical operations
- Code execution is sandboxed and validated
- Complex operations may require additional security checks
- Results depend on LLM code generation quality

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Example Queries

Here are some example queries you can try:

```python
# Area calculations
"Calculate the area of a circle given its radius is 5"
"Find the surface area of a cube given its side length is 3"

# Statistical operations
"Calculate the standard deviation of a list of numbers 1, 2, 3, 4, 5"
"Find the median of five numbers 1, 2, 3, 4, 5"

# Basic math
"Multiply three numbers together 2, 3, 4"
"Calculate the sum of squares for a list of numbers 1, 2, 3, 4, 5"
```

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure all dependencies are installed
```bash
pip install -r requirements.txt
```

2. **API Key Errors**: Verify your environment variables are set
```bash
echo $OPENAI_API_KEY  # Should show your API key
```

3. **Execution Errors**: Check that your query is clear and mathematical
```python
# Good query
calculator.calculate("Add 3 and 5 together")

# Bad query (non-mathematical)
calculator.calculate("Write a web server")  # Will fail
```

For more help, please open an issue on GitHub.

## TO-DO

- [ ] Add OpenAI provider support
- [ ] Add Azure OpenAI provider support
- [ ] Add more example queries and use cases
- [ ] Improve error handling and validation
- [ ] Add unit tests for different providers