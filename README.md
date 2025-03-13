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
```bash
# For OpenAI
export OPENAI_API_KEY=your_api_key_here

# For Anthropic (if using)
export ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

```python
from langchain_openai import ChatOpenAI
from calc_agent.calculator import CalculatorAgent
from calc_agent.llm_provider import OpenAIProvider

# Initialize with OpenAI
llm = ChatOpenAI()
provider = OpenAIProvider(llm)
calculator = CalculatorAgent(provider)

# Calculate area of a triangle
result = calculator.calculate(
    "Create a function to calculate the area of a triangle given base and height",
    [5, 3]
)
print(f"Area of triangle: {result}")  # Output: Area of triangle: 7.5

# Calculate average of numbers
result = calculator.calculate(
    "Write a function to find the average of three numbers",
    [10, 20, 30]
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
"Calculate the area of a circle given its radius"
"Find the surface area of a cube given its side length"

# Statistical operations
"Calculate the standard deviation of a list of numbers"
"Find the median of five numbers"

# Basic math
"Multiply three numbers together"
"Calculate the sum of squares for a list of numbers"
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
calculator.calculate("Add two numbers together", [5, 3])

# Bad query (non-mathematical)
calculator.calculate("Write a web server", [])  # Will fail
```

For more help, please open an issue on GitHub. 