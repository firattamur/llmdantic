class LLMdanticException(Exception):
    """Base class for all LLMdantic exceptions."""


class LLMdanticJsonParserException(LLMdanticException):
    """An exception raised when JSON parsing fails."""
