from triage.client import _trace_inputs, make_client
from triage.config import Settings

FAKE_KEY = "sk-or-v1-FAKEKEYFORTEST0000"


def test_api_key_never_reaches_trace_inputs():
    settings = Settings(
        model_id="some/model", temperature=0, max_tokens=10, base_url="https://example.invalid", api_key=FAKE_KEY
    )
    inputs = {
        "client": make_client(settings),
        "messages": [{"role": "user", "content": "hi"}],
        "model": settings.model_id,
        "temperature": 0,
        "max_tokens": 10,
    }
    traced = _trace_inputs(inputs)
    assert set(traced) == {"messages", "model", "temperature", "max_tokens"}
    assert FAKE_KEY not in repr(traced)
