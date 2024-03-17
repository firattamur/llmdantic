from typing import List, Optional

import pytest
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, field_validator

from llmdantic.output_parsers import LLMOutputParser
from llmdantic.prompts import LLMPromptBuilder

MOCK_OBJECTIVE = "Test Objective"


class MockInputNestedModel(BaseModel):
    nested1: str
    nested2: int

    @field_validator("nested1")
    def nested1_must_be_world(cls, v):
        """nested1 must be 'world'"""
        if v != "world":
            raise ValueError("nested1 must be 'world'")
        return v

    @field_validator("nested2")
    def nested2_must_be_negative(cls, v):
        """nested2 must be negative"""
        if v > 0:
            raise ValueError("nested2 must be negative")
        return v


class MockInputModel(BaseModel):
    input1: str
    input2: int
    input3: List[str]
    input4: Optional[str]
    input5: MockInputNestedModel

    @field_validator("input1")
    def input1_must_be_hello(cls, v):
        """input1 must be 'hello'"""
        if v != "hello":
            raise ValueError("input1 must be 'hello'")
        return v

    @field_validator("input2")
    def input2_must_be_positive(cls, v):
        """input2 must be positive"""
        if v < 0:
            raise ValueError("input2 must be positive")
        return v

    @field_validator("input3")
    def input3_must_be_non_empty(cls, v):
        """input3 must be non-empty"""
        if not v:
            raise ValueError("input3 must be non-empty")
        return v

    @field_validator("input4")
    def input4_must_be_none_or_empty(cls, v):
        """input4 must be None or empty"""
        if v and v != "":
            raise ValueError("input4 must be None or empty")
        return v


class MockOutputModelWithRules(BaseModel):
    output1: str
    output2: int
    output3: List[str]
    output4: Optional[str]

    @field_validator("output1")
    def output1_must_be_hello(cls, v):
        """output1 must be 'hello'"""
        if v != "hello":
            raise ValueError("output1 must be 'hello'")
        return v

    @field_validator("output2")
    def output2_must_be_positive(cls, v):
        """output2 must be positive"""
        if v < 0:
            raise ValueError("output2 must be positive")
        return v

    @field_validator("output3")
    def output3_must_be_non_empty(cls, v):
        """output3 must be non-empty"""
        if not v:
            raise ValueError("output3 must be non-empty")
        return v

    @field_validator("output4")
    def output4_must_be_none_or_empty(cls, v):
        """output4 must be None or empty"""
        if v and v != "":
            raise ValueError("output4 must be None or empty")
        return v


class MockOutputModelWithoutRules(BaseModel):
    output1: str
    output2: int
    output3: List[str]
    output4: Optional[str]


class MockParser(LLMOutputParser):
    def get_format_instructions(self):
        return "This is a mock format instruction."


@pytest.fixture
def llm_prompt_builder():
    inp_model = MockInputModel
    out_model = MockOutputModelWithRules
    parser = MockParser(pydantic_object=out_model)

    return LLMPromptBuilder(MOCK_OBJECTIVE, inp_model, out_model, parser)


def test_build_template_with_rules(llm_prompt_builder):
    llm_prompt_builder.out_model = MockOutputModelWithRules
    template = llm_prompt_builder.build_template()

    expected_template = """Objective: Test Objective

Input 'MockInputModel': 
{input}

Output 'MockOutputModelWithRules''s fields MUST FOLLOW the RULES:
MockOutputModelWithRules.output1:
• OUTPUT1 MUST BE 'HELLO'

MockOutputModelWithRules.output2:
• OUTPUT2 MUST BE POSITIVE

MockOutputModelWithRules.output3:
• OUTPUT3 MUST BE NON-EMPTY

MockOutputModelWithRules.output4:
• OUTPUT4 MUST BE NONE OR EMPTY

{format_instructions}
"""

    assert template.template == expected_template
    assert template.input_variables == ['input']
    assert template.partial_variables == {'format_instructions': 'This is a mock format instruction.'}


def test_build_template_no_rules(llm_prompt_builder):
    llm_prompt_builder.out_model = MockOutputModelWithoutRules
    template = llm_prompt_builder.build_template()

    expected_template = """Objective: Test Objective

Input 'MockInputModel': 
{input}

{format_instructions}
"""

    assert template.template == expected_template
    assert template.input_variables == ['input']
    assert template.partial_variables == {'format_instructions': 'This is a mock format instruction.'}


def test_build_prompt_with_rules(llm_prompt_builder):
    llm_prompt_builder.out_model = MockOutputModelWithRules
    template = llm_prompt_builder.build_template()

    input_data: MockInputModel = MockInputModel(
        input1="hello",
        input2=1,
        input3=["hello", "world"],
        input4=None,
        input5=MockInputNestedModel(nested1="world", nested2=-1),
    )
    actual_prompt = template.format(input=input_data.model_dump(mode='python'))

    expected_prompt = """Objective: Test Objective

Input 'MockInputModel': 
{'input1': 'hello', 'input2': 1, 'input3': ['hello', 'world'], 'input4': None, 'input5': {'nested1': 'world', 'nested2': -1}}

Output 'MockOutputModelWithRules''s fields MUST FOLLOW the RULES:
MockOutputModelWithRules.output1:
• OUTPUT1 MUST BE 'HELLO'

MockOutputModelWithRules.output2:
• OUTPUT2 MUST BE POSITIVE

MockOutputModelWithRules.output3:
• OUTPUT3 MUST BE NON-EMPTY

MockOutputModelWithRules.output4:
• OUTPUT4 MUST BE NONE OR EMPTY

This is a mock format instruction.
"""

    assert actual_prompt == expected_prompt


def test_build_prompt_no_rules(llm_prompt_builder):
    llm_prompt_builder.out_model = MockOutputModelWithoutRules
    template: PromptTemplate = llm_prompt_builder.build_template()

    input_data: MockInputModel = MockInputModel(
        input1="hello",
        input2=1,
        input3=["hello", "world"],
        input4=None,
        input5=MockInputNestedModel(nested1="world", nested2=-1),
    )
    actual_prompt: str = template.format(input=input_data.model_dump(mode='python'))

    expected_prompt: str = """Objective: Test Objective

Input 'MockInputModel': 
{'input1': 'hello', 'input2': 1, 'input3': ['hello', 'world'], 'input4': None, 'input5': {'nested1': 'world', 'nested2': -1}}

This is a mock format instruction.
"""

    assert actual_prompt == expected_prompt


def test_build_out_model_rules(llm_prompt_builder):
    llm_prompt_builder.out_model = MockOutputModelWithRules
    actual_rules: Optional[str] = llm_prompt_builder._build_out_model_rules()

    expected_rules: str = """MockOutputModelWithRules.output1:
• OUTPUT1 MUST BE 'HELLO'

MockOutputModelWithRules.output2:
• OUTPUT2 MUST BE POSITIVE

MockOutputModelWithRules.output3:
• OUTPUT3 MUST BE NON-EMPTY

MockOutputModelWithRules.output4:
• OUTPUT4 MUST BE NONE OR EMPTY"""

    assert actual_rules == expected_rules


def test_build_out_model_rules_no_rules(llm_prompt_builder):
    llm_prompt_builder.out_model = MockOutputModelWithoutRules
    actual_rules: Optional[str] = llm_prompt_builder._build_out_model_rules()

    assert actual_rules is None
