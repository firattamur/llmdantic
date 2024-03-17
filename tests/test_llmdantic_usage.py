from llmdantic.models import LLMdanticUsage


def test_llmdantic_result_total_tokens():
    result = LLMdanticUsage(
        inp_tokens=10,
        out_tokens=20,
        successful_requests=0,
        cost=0.0,
        successful_outputs=0,
    )

    assert result.total_tokens == 30
