"""The request deadline must cover prompt prefill, not only decode."""

import os
import pytest
from harness import Client, RunnerServer, find_runner

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture(scope="module")
def tiny_timeout_client(report):
    # A deadline this short has already passed when prefill starts, so the
    # first chunk boundary is where it must be noticed.
    with RunnerServer(find_runner(ROOT), os.path.join(ROOT, "test.gguf"),
                      ctx=2048, parallel=1,
                      # -b 32: the deadline is only observable at a chunk
                      # BOUNDARY (a single model_forward_batch is not
                      # interruptible), so the prompt has to span several
                      # batches. Forcing a small batch makes that true of a
                      # short prompt and keeps the test quick.
                      extra_args=["--gpu", "off", "-b", "32"],
                      env={"RUNNER_REQUEST_TIMEOUT": "0.001"}) as server:
        yield Client(server, report)


def test_prefill_respects_the_request_deadline(tiny_timeout_client):
    """The bound was computed before any work but handed only to the decode
    loop, so a long enough prompt overran its own timeout by the entire
    prefill. It must be reported as a timeout, not as a context overflow --
    the prompt fits, the clock ran out."""
    r = tiny_timeout_client.raw(
        "prefill-deadline", "POST", "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "word " * 60}],
         "max_tokens": 4, "temperature": 0})
    assert r.status == 408, (r.status, r.text[:300])
    assert "timed out" in r.text, r.text[:300]
