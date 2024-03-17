from typing import List

import pytest
from langchain_core.outputs import Generation
from pydantic import BaseModel

from llmdantic.output_parsers import LLMOutputParser


class MockNestedOutputModel(BaseModel):
    nested_output1: str
    nested_output2: int
    nested_output3: List


class MockOutputModel(BaseModel):
    output1: str
    output2: int
    output3: List[str]
    output4: List[MockNestedOutputModel]


@pytest.fixture
def llm_output_parser():
    return LLMOutputParser(pydantic_object=MockOutputModel)


def test_parse_result_when_result_is_empty(llm_output_parser):
    result = llm_output_parser.parse_result(result=[], partial=False)

    assert result.text == ""
    assert result.output is None


def test_parse_result_when_result_is_not_empty(llm_output_parser):
    result = llm_output_parser.parse_result(
        result=[
            Generation(
                text='{"output1": "hello", "output2": 1, "output3": ["output3"], "output4": [{"nested_output1": "nested_output1", "nested_output2": 1, "nested_output3": ["nested_output3"]}]}'
            )
        ],
        partial=False,
    )

    assert result.text == (
        '{"output1": "hello", "output2": 1, "output3": ["output3"], "output4": [{"nested_output1": "nested_output1", "nested_output2": 1, "nested_output3": ["nested_output3"]}]}'
    )
    assert result.output == MockOutputModel(
        output1="hello",
        output2=1,
        output3=["output3"],
        output4=[
            MockNestedOutputModel(
                nested_output1="nested_output1",
                nested_output2=1,
                nested_output3=["nested_output3"],
            )
        ],
    )


def test_parse_result_when_result_is_not_empty_and_invalid_json(llm_output_parser):
    result = llm_output_parser.parse_result(
        result=[Generation(text="This is an invalid JSON string.")], partial=False
    )

    assert result.text == "This is an invalid JSON string."
    assert result.output is None


def test_parse_result_when_result_is_not_empty_and_invalid_json_object(llm_output_parser):
    result = llm_output_parser.parse_result(
        result=[Generation(
            text="""'{"output1": "hello", "output2": 1, "output3": ["output3"], "output4": [{"nested_output1": "nested_output1", "nested_output2": 1, "nested_output3": ["nested_output3"]}]')]"""
        )],
    )

    assert result.text == """'{"output1": "hello", "output2": 1, "output3": ["output3"], "output4": [{"nested_output1": "nested_output1", "nested_output2": 1, "nested_output3": ["nested_output3"]}]')]"""
    assert result.output is None


def test_get_format_instructions(llm_output_parser):
    format_instructions = llm_output_parser.get_format_instructions()

    expected_format_instructions = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"$defs": {"MockNestedOutputModel": {"properties": {"nested_output1": {"title": "Nested Output1", "type": "string"}, "nested_output2": {"title": "Nested Output2", "type": "integer"}, "nested_output3": {"items": {}, "title": "Nested Output3", "type": "array"}}, "required": ["nested_output1", "nested_output2", "nested_output3"], "title": "MockNestedOutputModel", "type": "object"}}, "properties": {"output1": {"title": "Output1", "type": "string"}, "output2": {"title": "Output2", "type": "integer"}, "output3": {"items": {"type": "string"}, "title": "Output3", "type": "array"}, "output4": {"items": {"$ref": "#/$defs/MockNestedOutputModel"}, "title": "Output4", "type": "array"}}, "required": ["output1", "output2", "output3", "output4"]}
```"""

    assert format_instructions == expected_format_instructions


def test_parse_pydantic_object_from_text(llm_output_parser):
    text = '{"output1": "hello", "output2": 1, "output3": ["output3"], "output4": [{"nested_output1": "nested_output1", "nested_output2": 1, "nested_output3": ["nested_output3"]}]}'

    pydantic_object = llm_output_parser._parse_pydantic_object_from_text(text)

    assert pydantic_object == MockOutputModel(
        output1="hello",
        output2=1,
        output3=["output3"],
        output4=[
            MockNestedOutputModel(
                nested_output1="nested_output1",
                nested_output2=1,
                nested_output3=["nested_output3"],
            )
        ],
    )


def test_parse_pydantic_object_from_text_with_invalid_json(llm_output_parser):
    text = "This is an invalid JSON string."

    with pytest.raises(Exception) as exc_info:
        llm_output_parser._parse_pydantic_object_from_text(text)

    assert str(exc_info.value) == "Failed to find any JSON objects in the input text."


def test_parse_pydantic_object_from_text_with_invalid_json_object(llm_output_parser):
    text = '{"output1": "hello", "output2": 1, "output3": ["output3"], "output4": [{"nested_output1": "nested_output1", "nested_output2": 1, "nested_output3": ["nested_output3"]}]'

    with pytest.raises(Exception) as exc_info:
        llm_output_parser._parse_pydantic_object_from_text(text)

    assert str(exc_info.value) == "Failed to parse the output into a Pydantic model."
