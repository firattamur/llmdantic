from typing import Optional

from langchain.prompts import PromptTemplate
from pydantic import BaseModel

from llmdantic.core import get_all_nested_validator_requirements
from llmdantic.output_parsers import LLMOutputParser
from .format_instructions import FORMAT_INSTRUCTIONS, FORMAT_INSTRUCTIONS_NO_RULES


class LLMPromptBuilder:
    def __init__(
            self,
            objective: str,
            inp_model: type[BaseModel],
            out_model: type[BaseModel],
            parser: LLMOutputParser,
    ) -> None:
        self.objective: str = objective
        self.inp_model: type[BaseModel] = inp_model
        self.out_model: type[BaseModel] = out_model
        self.parser: LLMOutputParser = parser

    def build_template(self) -> PromptTemplate:
        """
        Constructs and returns a PromptTemplate instance, encapsulating the full prompt configuration
        for the language learning model, including the template string, input variables, and any partial
        variables required for the prompt's format instructions.

        Returns:
            PromptTemplate: An instance of PromptTemplate containing all necessary components for the LLM prompt.
        """
        return PromptTemplate(
            template=self._build_prompt(),
            input_variables=['input'],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    def _build_prompt(self) -> str:
        """
        Constructs the complete prompt string for the language learning model by integrating various components,
        such as the goal of the prompt, input and output model fields, specific rules for the output model, and
        schema or format instructions.

        This method formats a predefined template string (FORMAT_INSTRUCTIONS) with these components to create
        a cohesive and structured prompt ready for use with the LLM.

        Returns:
            str: The fully constructed prompt string.
        """
        if get_all_nested_validator_requirements(self.out_model):
            return FORMAT_INSTRUCTIONS.format(
                objective=self.objective,
                inp_model=self.inp_model.__name__,
                out_model=self.out_model.__name__,
                rules=self._build_out_model_rules(),
                schema=self.parser.get_format_instructions(),
            )

        return FORMAT_INSTRUCTIONS_NO_RULES.format(
            objective=self.objective,
            inp_model=self.inp_model.__name__,
            schema=self.parser.get_format_instructions(),
        )

    def _build_out_model_rules(self) -> Optional[str]:
        """
        Builds a descriptive string for the output model field rules section of the prompt,
        detailing the rules associated with each field in the output model.

        This method iterates over each field and its associated rules, formatting them
        into a readable string that lists all rules for each field, presented with bullet points
        and in uppercase for emphasis.

        Returns:
            str: A formatted string listing all fields and their associated rules, with each
                rule presented in a bullet point list and in uppercase.
        """
        formatted_rules = []
        requirements = get_all_nested_validator_requirements(self.out_model)

        if not requirements:
            return None

        for field, rules in requirements.items():
            formatted_field_rules = [f"{field}:"]

            for rule in rules:
                formatted_field_rules.append(f"• {rule.upper()}")

            formatted_rules.append("\n".join(formatted_field_rules))

        return "\n\n".join(formatted_rules)
