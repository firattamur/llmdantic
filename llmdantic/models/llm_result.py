from typing import Optional

from pydantic import BaseModel, Field


class LLMdanticResult(BaseModel):
    """Result model for the LLMdantic class."""

    text: str = Field(..., description="Generated text by llm.")
    output: Optional[BaseModel] = Field(
        None, description="Output model parsed from generated text."
    )
    retry_count: int = Field(
        1, description="The number of times the processing was retried."
    )
    cost: float = Field(
        0.0, description="The cost of the processing in USD.", ge=0.0
    )
    inp_tokens: int = Field(
        0, description="The number of tokens in the input prompt.", ge=0
    )
    out_tokens: int = Field(
        0, description="The number of tokens in the generated text.", ge=0
    )
    successful_requests: int = Field(
        0, description="The number of successful requests made to the LLM.", ge=0
    )

    @property
    def total_tokens(self):
        return self.inp_tokens + self.out_tokens

    def __str__(self):
        return (
            "LLMdanticResult(\n"
            f"  text={self.text},\n"
            f"  output={self.output},\n"
            f"  retry_count={self.retry_count},\n"
            f"  cost={self.cost:.f},\n"
            f"  inp_tokens={self.inp_tokens},\n"
            f"  out_tokens={self.out_tokens},\n"
            f"  total_tokens={self.total_tokens},\n"
            f"  successful_requests={self.successful_requests}\n"
            ")"
        )
