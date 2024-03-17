from typing import Type

from pydantic import BaseModel, Field

LLMSchema = Type[BaseModel]


class LLMdanticConfig(BaseModel):
    """A model for the LLMdantic configuration."""

    objective: str = Field(
        ...,
        description="The objective of the LLM. It is used to construct the prompt for the LLM.",
    )
    inp_schema: LLMSchema = Field(
        ...,
        description="The input model for the LLM. It is used to construct the prompt for the LLM.",
    )
    out_schema: LLMSchema = Field(
        ...,
        description="The output model for the LLM. It is used to validate the output of the LLM.",
    )
    retry: int = Field(
        1,
        description="The number of times to retry the interaction in case of failure.", ge=1
    )
    verbose: bool = Field(
        False,
        description="Whether to print verbose logs.",
    )
