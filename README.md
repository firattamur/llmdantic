<br>

<p align="center">
  <picture> 
    <img src="https://github.com/firattamur/llmdantic/blob/main/.github/assets/llmdantic.png" alt="image" width="300">
  </picture>
</p>

<h3 style="font-size: 5em" align="center">
    Structured Output Is All You Need!
</h3>

<br>

LLMdantic redefines the integration of Large Language Models (LLMs) into your projects, offering a seamless, efficient,
and powerful way to work with the latest advancements in AI. By abstracting the complexities of LLMs, llmdantic allows
developers to focus on what truly matters: building innovative applications.

* **Ease of Use.** Simplify your LLM interactions. Forget about the intricacies of prompts and models; define your
  requirements and let llmdantic handle the rest.
* **Data Integrity.** With Pydantic, define input and output models that ensure your data aligns perfectly with your
  requirements, maintaining structure and validation at every step.
* **Modular and Extensible.** Easily switch between different LLMs and customize your experience with the modular and
  extensible framework provided by llmdantic.
* **Cost Tracking.** Keep track of your LLM usage and costs, ensuring you stay within budget and optimize your usage.
* **Batch Processing.** Process multiple data points in a single call, streamlining your operations and enhancing
  efficiency.
* **Retry Mechanism.** Automatically retry failed requests, ensuring you get the results you need without any hassle.

## Getting Started 🌟

### Installation

```bash
pip install llmdantic
```

### Usage

1. **Define Your Models**

- **inp_model**: Define the structure of the data you want to process.
- **out_model**: Define the structure of the data you expect to receive.
    - Use Pydantic to define your models and add custom validation rules.
    - Custom validation rules are used to ensure the integrity and quality of your data.
    - Add docstrings to your custom validation rules to provide prompts for the LLM.

```python
from pydantic import BaseModel, field_validator


class SummarizeInput(BaseModel):
    text: str


class SummarizeOutput(BaseModel):
    summary: str

    @field_validator("summary")
    def summary_must_not_be_empty(cls, v) -> bool:
        """Summary cannot be empty"""  # Add docstring that explains the validation rule. This will be used as a prompt for the LLM
        if not v.strip():
            raise
        return v

    @field_validator("summary")
    def summary_must_be_short(cls, v) -> bool:
        """Summary must be less than 100 words"""  # Add docstring that explains the validation rule. This will be used as a prompt for the LLM
        if len(v.split()) > 100:
            raise
        return v
```

2. **Initialize LLMdantic**

- Initialize **LLMdantic** with your input and output models.
- Also, provide a objective for the LLM to understand the task.

```python
from llmdantic import LLMdantic, LLMdanticConfig
from langchain_openai import OpenAI
from langchain.llms.base import BaseLanguageModel

llm: BaseLanguageModel = OpenAI()

config: LLMdanticConfig = LLMdanticConfig(
    objective="Summarize the text",
    inp_model=SummarizeInput,
    out_model=SummarizeOutput,
    retries=3
)

llmdantic = LLMdantic(
    llm=llm,
    config=config
)
```

3. **Process Your Data**

- Use the `invoke` or `batch` method to process your data.

- `invoke` returns an instance of `LLMdanticResult` which contains:
    - `text`: The output of the LLM.
    - `output`: The output model with the processed data.
    - `retry_count`: The number of retries made to get the result.
    - `cost`: The cost of the request.
    - `inp_tokens`: The number of tokens used for the input.
    - `out_tokens`: The number of tokens used for the output.
    - `successful_requests`: The number of successful requests made.

- `batch` returns a list of `LLMdanticResult` for each input data.

```python
from llmdantic import LLMdanticResult

data: SummarizeInput = SummarizeInput(text="A long article about natural language processing...")
result: LLMdanticResult = llmdantic.invoke(data)

if result.output:
    print(result.output.summary)
```

- For batch processing, pass a list of input data.

```python
data: List[SummarizeInput] = [
    SummarizeInput(text="A long article about natural language processing..."),
    SummarizeInput(text="A long article about computer vision...")
]
results: List[Optional[SummarizeOutput]] = llmdantic.batch(data)

for result in results:
    if result:
        print(result.summary)
```

4. **Track Costs and Stats**:

- Use the `cost` attribute of the `LLMdanticResult` to track the cost of the request.
- Use the `usage` attribute of the `LLMdantic` to track the usage stats overall.

```python
from llmdantic import LLMdanticResult

data: SummarizeInput = SummarizeInput(text="A long article about natural language processing...")
result: LLMdanticResult = llmdantic.invoke(data)

if result.output:
    print(result.output.summary)

# Track the cost of the request
print(f"Cost: {result.cost}")

# Track the usage stats
print(f"Usage: {llmdantic.usage}")
```

## Advanced Usage 🛠

**LLMdantic** is built on top of the `langchain` package, which provides a modular and extensible framework for working
with LLMs. You can easily switch between different LLMs and customize your experience.

### Switching LLMs:

- **LLMdantic** uses the `OpenAI` LLM by default. You can switch to a different LLM by providing an instance of the
  desired LLM.

```python
from llmdantic import LLMdantic, LLMdanticConfig
from langchain_community.llm.ollama import Ollama
from langchain.llms.base import BaseLanguageModel

llm: BaseLanguageModel = Ollama()

config: LLMdanticConfig = LLMdanticConfig(
    objective="Summarize the text",
    inp_model=SummarizeInput,
    out_model=SummarizeOutput,
    retries=3
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
