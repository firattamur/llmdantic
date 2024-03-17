from pydantic import BaseModel, Field


class LLMdanticUsage(BaseModel):
    """Usage model for the LLMdantic class."""

    inp_tokens: int = Field(
        0, description="The number of tokens in the input prompt.", ge=0
    )

    out_tokens: int = Field(
        0, description="The number of tokens in the generated text.", ge=0
    )

    successful_requests: int = Field(
        0, description="The number of successful requests made to the LLM.", ge=0
    )

    cost: float = Field(
        0.0, description="The cost of the processing in USD.", ge=0.0
    )

    successful_outputs: int = Field(
        0, description="The number of successful outputs.", ge=0
    )

    @property
    def total_tokens(self):
        return self.inp_tokens + self.out_tokens

    def __str__(self):
        return (
            "LLMdanticUsage(\n"
            f"  inp_tokens={self.inp_tokens},\n"
            f"  out_tokens={self.out_tokens},\n"
            f"  total_tokens={self.total_tokens},\n"
            f"  successful_requests={self.successful_requests},\n"
            f"  cost={self.cost:.6f},\n"
            f"  successful_outputs={self.successful_outputs}\n"
            ")"
        )
