import json
from collections import defaultdict
from typing import Dict, List, Any
from typing import Optional, get_origin, get_args

import regex
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai.chat_models.base import ChatOpenAI
from pydantic import BaseModel
from regex import Pattern

from llmdantic.core import LLMdanticJsonParserException
from llmdantic.models import LLMSchema


def get_llm_callback_handler(llm: BaseLanguageModel) -> Optional[BaseCallbackHandler]:
    """
    Get the LLM callback handler for the given LLM class.

    Args:
        llm (BaseLanguageModel): The LLM class to get the callback handler for.

    Returns:
        Optional[BaseCallbackHandler]: The LLM callback handler if available, None otherwise.
    """
    if issubclass(llm.__class__, ChatOpenAI):
        return OpenAICallbackHandler()

    return None


def parse_json_objects_from_text(text: str) -> List[Dict]:
    """
    Parse JSON objects from a string and return a list of dictionaries.

    Args:
        text (str): The string containing JSON objects to parse.

    Returns:
        List[Dict]: A list of dictionaries parsed from the JSON objects found in the input string.

    Raises:
        LLMdanticJsonParserException: If no JSON objects can be parsed from the input string.
    """
    json_pattern: Pattern = regex.compile(
        r"(?:\[(?:[^\[\]]|(?R))*\]|\{(?:[^{}]|(?R))*\})"
    )

    matches = json_pattern.findall(text.strip())

    if not matches:
        raise LLMdanticJsonParserException(
            "Failed to find any JSON objects in the input text."
        )

    json_objects = []
    for match in matches:
        try:
            json_objects.append(json.loads(match))
        except json.JSONDecodeError:
            continue

    if not json_objects:
        raise LLMdanticJsonParserException(
            "Failed to find any JSON objects in the input text."
        )

    return json_objects


def get_model_field_info(cls: LLMSchema) -> List[Dict[str, Any]]:
    """
    Retrieve all field names with their respective types.

    Args:
        cls (LLMSchema): The Pydantic model class.

    Returns:
        List[Dict[str, Any]]: A list containing all field names with their respective types.
    """
    return [
        {"name": field_name, "type": field_type.annotation}
        for field_name, field_type in cls.model_fields.items()
    ]


def get_model_field_names(cls: LLMSchema) -> Dict[str, str]:
    """
    Retrieve all field names with their respective types as key and field name as value.

    Args:
        cls (LLMSchema): The Pydantic model class.

    Returns:
        Dict[str, str]: A dictionary containing all field types as key and field name as value.
    """
    field_names = {}

    for field in get_model_field_info(cls):
        key = field["name"]
        field_names[key] = key

    return field_names


def get_field_validator_requirements(cls: LLMSchema) -> Dict[str, List[str]]:
    """
    Collect and return the requirements for all fields within the class.

    Args:
        cls (LLMSchema): The Pydantic model class.

    Returns:
        Dict[str, List[str]]: A dictionary containing field names and their respective requirements.
    """
    requirements = defaultdict(list)

    for _, validator_info in cls.__pydantic_decorators__.field_validators.items():
        fields = validator_info.info.fields
        requirement = validator_info.func.__doc__

        for field_name in fields:
            requirements[f"{cls.__name__}.{field_name}"].append(requirement)

    return dict(requirements)


def get_all_nested_validator_requirements(cls: LLMSchema) -> Dict[str, List[str]]:
    """
    Recursively collect and return the requirements for all fields within the class.

    Args:
        cls (LLMSchema): The Pydantic model class.

    Returns:
        Dict[str, List[str]]: A dictionary containing field names and their respective requirements.
    """
    requirements = get_field_validator_requirements(cls)

    for field in get_model_field_info(cls):
        field_type = field["type"]
        field_origin = get_origin(field_type)

        if field_origin:
            field_type = get_args(field_type)[0]

        if issubclass(field_type, BaseModel):
            requirements.update(get_all_nested_validator_requirements(field_type))

    return requirements
