<p align="center">
 <picture>
   <img src="https://raw.githubusercontent.com/firattamur/llmdantic/main/.github/assets/llmdantic.png" alt="image" width="300">
 </picture>  
</p>

<h3 style="font-size: 5em" align="center">
   Structured Output Is All You Need!  
</h3>

<br>

LLMdantic is a powerful and efficient Python library that simplifies the integration of Large Language Models (LLMs) into your projects. Built on top of the incredible [Langchain](https://github.com/hwchase17/langchain) package and leveraging the power of [Pydantic](https://github.com/pydantic/pydantic) models, LLMdantic provides a seamless and structured approach to working with LLMs.

## Features 🚀

- 🌐 Wide range of LLM support through Langchain integrations
- 🛡️ Ensures data integrity with Pydantic models for input and output validation
- 🧩 Modular and extensible design for easy customization
- 💰 Cost tracking and optimization for OpenAI models
- 🚀 Efficient batch processing for handling multiple data points
- 🔄 Robust retry mechanism for smooth and uninterrupted experience

## Getting Started 🌟

### Requirements

Before using LLMdantic, make sure you have set the required API keys for the LLMs you plan to use. For example, if you're using OpenAI's models, set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="your-api-key"
```

If you're using other LLMs, follow the instructions provided by the respective providers in Langchain's documentation.

### Installation

```bash
pip install llmdantic
```

### Usage

#### 1. Define input and output schemas using Pydantic:

- Use Pydantic to define input and output models with custom validation rules.

> [!IMPORTANT]
>
> Add docstrings to validation rules to provide prompts for the LLM. This will help the LLM understand the validation rules and provide better results


```python
from pydantic import BaseModel, field_validator

class SummarizeInput(BaseModel):
    text: str

class SummarizeOutput(BaseModel):
    summary: str

    @field_validator("summary")  
    def summary_must_not_be_empty(cls, v) -> bool:
        """Summary cannot be empty"""  # Add docstring that explains the validation rule. This will be used as a prompt for the LLM.
        if not v.strip():
            raise
        return v

    @field_validator("summary")
    def summary_must_be_short(cls, v) -> bool:  
        """Summary must be less than 100 words"""  # Add docstring that explains the validation rule. This will be used as a prompt for the LLM.
        if len(v.split()) > 100:
            raise  
        return v
```

#### 2. Create an LLMdantic client:

- Provide input and output models, objective, and configuration.

> [!TIP]
>
> The `objective` is a prompt that will be used to generate the actual prompt sent to the LLM. It should be a high-level description of the task you want the LLM to perform.
>
> The `inp_schema` and `out_schema` are the input and output models you defined in the previous step.
> 
> The `retries` parameter is the number of times the LLMdantic will retry the request in case of failure.

```python
from llmdantic import LLMdantic, LLMdanticConfig  
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

config: LLMdanticConfig = LLMdanticConfig(
    objective="Summarize the text", 
    inp_schema=SummarizeInput,
    out_schema=SummarizeOutput, 
    retries=3,
)

llmdantic = LLMdantic(llm=llm, config=config)
```

Here's the prompt template generated based on the input and output models:

```text
Objective: Summarize the text

Input 'SummarizeInput': 
{input}

Output 'SummarizeOutput''s fields MUST FOLLOW the RULES:
SummarizeOutput.summary:
• SUMMARY CANNOT BE EMPTY
• SUMMARY MUST BE LESS THAN 100 WORDS

{format_instructions}
```

#### 3. Generate output using the LLMdantic:

> [!TIP]
>
> The `invoke` method is used for single requests, while the `batch` method is used for batch processing.
>
> The `invoke` method returns an instance of `LLMdanticResult`, which contains the generated text, parsed output, and other useful information such as cost and usage stats such as the number of input and output tokens. Check out the [LLMdanticResult](#LLMdanticResult) model for more details.
>

```python
from llmdantic import LLMdanticResult

data = SummarizeInput(text="A long article about natural language processing...")
result: LLMdanticResult = llmdantic.invoke(data)

output: Optional[SummarizeOutput] = result.output

if output:
    print(output.summary)
```

Here's the actual prompt sent to the LLM based on the input data:

```text
Objective: Summarize the text

Input 'SummarizeInput': 
{'text': 'A long article about natural language processing...'}

Output 'SummarizeOutput''s fields MUST FOLLOW the RULES:
SummarizeOutput.summary:
• SUMMARY CANNOT BE EMPTY
• SUMMARY MUST BE LESS THAN 100 WORDS

The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
{"properties": {"summary": {"title": "Summary", "type": "string"}}, "required": ["summary"]}
```

- For batch processing, pass a list of input data.

> [!IMPORTANT]
>
> The `batch` method returns a list of `LLMdanticResult` instances, each containing the generated text, parsed output, and other useful information such as cost and usage stats such as the number of input and output tokens. Check out the [LLMdanticResult](#LLMdanticResult) model for more details.
>
> The `concurrency` parameter is the number of concurrent requests to be made. Please check the usage limits of the LLM provider before setting this value.
>

```python
data: List[SummarizeInput] = [
    SummarizeInput(text="A long article about natural language processing..."),
    SummarizeInput(text="A long article about computer vision...")  
]
results: List[LLMdanticResult] = llmdantic.batch(data, concurrency=2)

for result in results:
    if result.output:
        print(result.output.summary)
```

#### 4. Monitor usage and costs:

> [!IMPORTANT]
>
> The cost tracking feature is currently available for OpenAI models only.
>
> The `usage` attribute returns an instance of `LLMdanticUsage`, which contains the number of input and output tokens, successful requests, cost, and successful outputs. Check out the [LLMdanticUsage](#LLMdanticUsage) model for more details.
>
> Please note that the usage is tracked for the entire lifetime of the `LLMdantic` instance. 

- Use the `cost` attribute of the LLMdanticResult to track the cost of the request (currently available for OpenAI models).

- Use the `usage` attribute of the LLMdantic to track the usage stats overall.

```python
from llmdantic import LLMdanticResult

data: SummarizeInput = SummarizeInput(text="A long article about natural language processing...")  
result: LLMdanticResult = llmdantic.invoke(data)

if result.output:
    print(result.output.summary)

# Track the cost of the request (OpenAI models only)
print(f"Cost: {result.cost}")  

# Track the usage stats
print(f"Usage: {llmdantic.usage}")
```

```bash
Cost: 0.0003665
Overall Usage: LLMdanticUsage(
  inp_tokens=219,
  out_tokens=19,
  total_tokens=238,
  successful_requests=1,
  cost=0.000367,
  successful_outputs=1
)
```

## Advanced Usage 🛠

`LLMdantic` is built on top of the langchain package, which provides a modular and extensible framework for working with LLMs. You can easily switch between different LLMs and customize your experience.

Switching LLMs

> [!IMPORTANT]
>
> Make sure to set the required API keys for the new LLM you plan to use.
>
> The `llm` parameter of the `LLMdantic` class should be an instance of `BaseLanguageModel` from the langchain package.
> 

> [!TIP]
>
> You can use the `langchain_community` package to access a wide range of LLMs from different providers.
>
> You may need to provide model_name, api_key, and other parameters based on the LLM you want to use. Check out the documentation of the respective LLM provider for more details.
> 


```python
from llmdantic import LLMdantic, LLMdanticConfig
from langchain_community.llm.ollama import Ollama
from langchain.llms.base import BaseLanguageModel

llm: BaseLanguageModel = Ollama()

config: LLMdanticConfig = LLMdanticConfig(
    objective="Summarize the text",
    inp_schema=SummarizeInput, 
    out_schema=SummarizeOutput,
    retries=3,
)

llmdantic = LLMdantic(
    llm=llm,
    config=config
)
```

## Contributing 🤝

Contributions are welcome! Whether you're fixing bugs, adding new features, or improving documentation, your help makes
**LLMdantic** better for everyone. Feel free to open an issue or submit a pull request.

## License 📄

**LLMdantic** is released under the [MIT License](LICENSE). Feel free to use it, contribute, and spread the word!

