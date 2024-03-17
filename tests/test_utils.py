from unittest.mock import Mock

import pytest
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai.chat_models.base import ChatOpenAI
from pydantic import BaseModel, field_validator

from llmdantic.core import get_llm_callback_handler, parse_json_objects_from_text, LLMdanticJsonParserException
from llmdantic.core import (
    get_model_field_info, get_model_field_names,
    get_field_validator_requirements,
    get_all_nested_validator_requirements
)


def test_get_llm_callback_handler_when_llm_is_openai() -> None:
    llm_mock: BaseLanguageModel = Mock(spec=ChatOpenAI)
    callback_handler: BaseCallbackHandler = get_llm_callback_handler(llm_mock)

    assert callback_handler is not None
    assert isinstance(callback_handler, OpenAICallbackHandler)


def test_get_llm_callback_handler_when_llm_is_not_openai() -> None:
    llm_mock: BaseLanguageModel = Mock()
    callback_handler: BaseCallbackHandler = get_llm_callback_handler(llm_mock)

    assert callback_handler is None


def test_parse_json_objects_from_text_with_single_valid_json_object():
    text = 'This is a test string. {"key": "value"} Another string.'

    expected_output = [{'key': 'value'}]
    result = parse_json_objects_from_text(text)

    assert result == expected_output


def test_parse_json_objects_from_text_with_multiple_valid_json_objects():
    text = 'This is a test string. {"key": "value"} Another string. [{"a": 1}, {"b": 2}]'

    expected_output = [{'key': 'value'}, [{'a': 1}, {'b': 2}]]
    result = parse_json_objects_from_text(text)

    assert result == expected_output


def test_parse_json_objects_from_text_with_no_json_objects():
    text = "This is a test string without any JSON."

    with pytest.raises(LLMdanticJsonParserException) as exc_info:
        parse_json_objects_from_text(text)

    assert str(exc_info.value) == "Failed to find any JSON objects in the input text."


def test_parse_json_objects_from_text_with_invalid_json_objects():
    text = "This is a test string. {'key': 'value', Another string."

    with pytest.raises(LLMdanticJsonParserException) as exc_info:
        parse_json_objects_from_text(text)

    assert str(exc_info.value) == "Failed to find any JSON objects in the input text."


def test_parse_json_objects_from_text_with_empty_string():
    text = ""

    with pytest.raises(LLMdanticJsonParserException) as exc_info:
        parse_json_objects_from_text(text)

    assert str(exc_info.value) == "Failed to find any JSON objects in the input text."


def test_parse_json_objects_from_text_with_leading_and_trailing_whitespace():
    text = '  \n  {"key": "value"}  \n  '

    expected_output = [{'key': 'value'}]
    result = parse_json_objects_from_text(text)

    assert result == expected_output


def test_parse_json_objects_from_text_with_nested_json_objects():
    text = 'This is a test string. {"key": {"nested_key": "value"}} Another string.'

    expected_output = [{'key': {'nested_key': 'value'}}]
    result = parse_json_objects_from_text(text)

    assert result == expected_output


def test_get_field_info():
    class TestModel(BaseModel):
        a: int
        b: str
        c: float

    expected_output = [
        {"name": "a", "type": int},
        {"name": "b", "type": str},
        {"name": "c", "type": float},
    ]
    result = get_model_field_info(TestModel)

    assert result == expected_output


def test_get_field_info_with_nested_model():
    class NestedModel(BaseModel):
        a: int
        b: str
        c: float

    class TestModel(BaseModel):
        a: int
        b: str
        c: NestedModel

    expected_output = [
        {"name": "a", "type": int},
        {"name": "b", "type": str},
        {"name": "c", "type": NestedModel},
    ]
    result = get_model_field_info(TestModel)

    assert result == expected_output


def test_get_field_names():
    class TestModel(BaseModel):
        a: int
        b: str
        c: float

    expected_output = {
        "a": "a",
        "b": "b",
        "c": "c",
    }
    result = get_model_field_names(TestModel)

    assert result == expected_output


def test_get_field_names_with_nested_model():
    class NestedModel(BaseModel):
        a: int
        b: str
        c: float

    class TestModel(BaseModel):
        a: int
        b: str
        c: NestedModel

    expected_output = {
        "a": "a",
        "b": "b",
        "c": "c",
    }
    result = get_model_field_names(TestModel)

    assert result == expected_output


def test_get_field_requirements():
    class TestModel(BaseModel):
        a: int
        b: str
        c: float

        @field_validator("a")
        def a_must_be_positive(cls, v):
            """a must be positive"""
            assert v > 0, "a must be positive"

        @field_validator("b")
        def b_must_be_lowercase(cls, v):
            """b must be lowercase"""
            assert v.islower(), "b must be lowercase"

        @field_validator("c")
        def c_must_be_negative(cls, v):
            """c must be negative"""
            assert v < 0, "c must be negative"

    expected_output = {
        "TestModel.a": ["a must be positive"],
        "TestModel.b": ["b must be lowercase"],
        "TestModel.c": ["c must be negative"],
    }
    result = get_field_validator_requirements(TestModel)

    assert result == expected_output


def test_get_field_requirements_with_nested_model():
    class NestedModel(BaseModel):
        a: int
        b: str
        c: float

        @field_validator("a")
        def a_must_be_positive(cls, v):
            """a must be positive"""
            assert v > 0, "a must be positive"

        @field_validator("b")
        def b_must_be_lowercase(cls, v):
            """b must be lowercase"""
            assert v.islower(), "b must be lowercase"

        @field_validator("c")
        def c_must_be_negative(cls, v):
            """c must be negative"""
            assert v < 0, "c must be negative"

    class TestModel(BaseModel):
        a: int
        b: str
        c: NestedModel

    expected_output = {
        "NestedModel.a": ["a must be positive"],
        "NestedModel.b": ["b must be lowercase"],
        "NestedModel.c": ["c must be negative"],
    }
    result = get_all_nested_validator_requirements(TestModel)

    assert result == expected_output
