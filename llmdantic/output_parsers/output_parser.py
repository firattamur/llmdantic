import json
from typing import Dict, List

from langchain.output_parsers.format_instructions import PYDANTIC_FORMAT_INSTRUCTIONS
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import Generation
from pydantic import BaseModel, ValidationError

from llmdantic.core import LLMdanticJsonParserException
from llmdantic.core.utils import parse_json_objects_from_text
from llmdantic.models.llm_result import LLMdanticResult


class LLMOutputParser(JsonOutputParser):
    """A parser for the output of the LLM."""

    pydantic_object: type[BaseModel]

    def parse_result(
            self, result: List[Generation], *, partial: bool = False
    ) -> LLMdanticResult:
        """Parse the output of the LLM.

        Args:
            result (List[Generation]): The output of the LLM as a list of Generation objects.
            partial (bool, optional): Whether to parse the output partially. Defaults to False.

        Returns:
            LLMdanticResult: The parsed output of the LLM with the text and the output model.
        """
        if not result:
            return LLMdanticResult(text="", output=None)

        text: str = result[0].text

        try:
            output: BaseModel = self._parse_pydantic_object_from_text(text)
            return LLMdanticResult(text=text, output=output)

        except LLMdanticJsonParserException:
            return LLMdanticResult(text=text, output=None)

        except OutputParserException:
            return LLMdanticResult(text=text, output=None)

    def get_format_instructions(self) -> str:
        """Get the format instructions for the Pydantic model.

        Returns:
            str: The format instructions for the Pydantic model.
        """
        schema = self.pydantic_object.model_json_schema()

        reduced_schema = schema

        for key in ["title", "type"]:
            schema.pop(key, None)

        schema_str = json.dumps(reduced_schema)
        return PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    def _parse_pydantic_object_from_text(self, text: str) -> BaseModel:
        """Parse the output of the LLM into a Pydantic model.

        Args:
            text (str): The output of the LLM as a string.

        Returns:
            BaseModel: The parsed output of the LLM as a Pydantic model.
        """
        json_objects: List[Dict] = parse_json_objects_from_text(text)

        for json_object in json_objects:
            try:
                return self.pydantic_object.model_validate(json_object)

            except ValidationError:
                continue

        raise OutputParserException("Failed to parse the output into a Pydantic model.")
