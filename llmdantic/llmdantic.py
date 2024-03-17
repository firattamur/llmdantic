"""LLMdantic is a Python package for quickly and easily use LLMs to process any kind of data."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables.base import Runnable
from pydantic import BaseModel

from llmdantic.core import get_llm_callback_handler
from llmdantic.models import LLMdanticConfig, LLMdanticResult, LLMdanticUsage
from llmdantic.output_parsers import LLMOutputParser
from llmdantic.prompts import LLMPromptBuilder


class LLMdantic:
    def __init__(
            self,
            llm: BaseLanguageModel,
            config: LLMdanticConfig,
    ) -> None:
        """Initialize the LLMdantic object.

        Args:
            llm (BaseLanguageModel): The language model to use.
            config (LLMdanticConfig): The configuration object to use.
        """
        self.llm: BaseLanguageModel = llm
        self.config: LLMdanticConfig = config
        self.successful_outputs: int = 0

        self.output_parser: LLMOutputParser = LLMOutputParser(
            pydantic_object=config.out_schema
        )
        self.prompt_builder: LLMPromptBuilder = LLMPromptBuilder(
            config.objective, config.inp_schema, config.out_schema, self.output_parser
        )
        self.prompt_template: PromptTemplate = self.prompt_builder.build_template()

        self.chain: Runnable = self.prompt_template | self.llm | self.output_parser

        self.callback_handler: BaseCallbackHandler = get_llm_callback_handler(self.llm)
        self.callbacks = [self.callback_handler] if self.callback_handler else []

    def invoke(self, data: BaseModel) -> LLMdanticResult:
        """Process the given data.

        Args:
            data (BaseModel): The input data to be processed.

        Returns:
            LLMdanticResult: The processed data with the output model.
        """
        result: LLMdanticResult = LLMdanticResult(text="", output=None, retry_count=self.config.retry)

        for retry_count in range(1, self.config.retry + 1):
            result: LLMdanticResult = self._invoke(data)

            if result.output:
                result.retry_count = retry_count
                self.successful_outputs += 1

                break

        return result

    def batch(self, data: List[BaseModel], concurrency: int = 1) -> List[LLMdanticResult]:
        """Process a list of data objects in parallel.

        Args:
            data (List[BaseModel]): A list of data to be processed.
            concurrency (int, optional): The number of concurrent requests to make. Defaults to 1.

        Returns:
            List[BaseModel]: A list of processed data objects.
        """
        results = []
        batch_size: int = max(1, len(data) // concurrency)

        def process_batch(batch: List[BaseModel]) -> List[LLMdanticResult]:
            return [self.invoke(datum) for datum in batch]

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process_batch, data[i:i + batch_size]) for i in range(0, len(data), batch_size)]

            for future in as_completed(futures):
                batch_results = future.result()
                results.extend(batch_results)

        return results

    def _invoke(self, data: BaseModel) -> LLMdanticResult:
        """Invoke the LLM with the given data.

        Args:
            data (BaseModel): The data to be processed.

        Returns:
            LLMdanticResult: The processed data.
        """
        invoke_callback = get_llm_callback_handler(self.llm)
        if invoke_callback:
            self.callbacks.append(invoke_callback)

        result: LLMdanticResult = self.chain.invoke(
            {"input": data.dict()}, config={"callbacks": self.callbacks}
        )

        if invoke_callback:
            result.cost = invoke_callback.total_cost
            result.inp_tokens = invoke_callback.prompt_tokens
            result.out_tokens = invoke_callback.completion_tokens
            result.successful_requests = invoke_callback.successful_requests

            self.callbacks.remove(invoke_callback)

        return result

    def prompt(self, data: BaseModel) -> str:
        """Get the prompt for the given data.

        Args:
            data (BaseModel): The input data to be processed.

        Returns:
            str: The prompt for the given data.
        """
        return self.prompt_template.format(input=data.dict())

    @property
    def template(self) -> str:
        """Get the prompt template.

        Returns:
            str: The prompt template without the input data.
        """
        return self.prompt_template.template

    @property
    def usage(self) -> Optional[LLMdanticUsage]:
        """Get the usage of the LLMdantic object.

        Returns:
            Optional[LLMdanticUsage]: The usage of the LLMdantic object.
        """
        if not self.callback_handler:
            return None

        return LLMdanticUsage(
            inp_tokens=self.callback_handler.prompt_tokens,
            out_tokens=self.callback_handler.completion_tokens,
            successful_requests=self.callback_handler.successful_requests,
            cost=self.callback_handler.total_cost,
            successful_outputs=self.successful_outputs,
        )
