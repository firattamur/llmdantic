from llmdantic.models import LLMdanticResult


def test_llmdantic_result_total_tokens():
    result = LLMdanticResult(
        text="text",
        output=None,
        retry_count=1,
        cost=0.0,
        inp_tokens=10,
        out_tokens=20,
        successful_requests=0,
    )

    assert result.total_tokens == 30
