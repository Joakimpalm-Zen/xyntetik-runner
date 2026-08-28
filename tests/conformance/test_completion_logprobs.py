"""Legacy Completions logprobs used by reference-differential tooling."""

import os

import pytest

from harness import Client, RunnerServer, find_runner


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture(scope="module")
def completion_validation_client(report):
    with RunnerServer(find_runner(ROOT), os.path.join(ROOT, "test.gguf"),
                      ctx=64, parallel=1,
                      extra_args=["--gpu", "off"]) as server:
        yield Client(server, report)


def test_text_completion_rejects_unsupported_echo(completion_validation_client):
    completion_validation_client.expect_400({
        "prompt": "hello",
        "max_tokens": 1,
        "echo": True,
    }, name="completion-echo-unsupported", contains="echo",
        path="/v1/completions")


def test_text_completion_rejects_unsupported_prompt_logprobs(
        completion_validation_client):
    completion_validation_client.expect_400({
        "prompt": "hello",
        "max_tokens": 1,
        "prompt_logprobs": 1,
    }, name="completion-prompt-logprobs-unsupported",
        contains="prompt_logprobs", path="/v1/completions")


def test_text_completion_returns_requested_logprobs(client):
    response = client.completion({
        "prompt": "hello",
        "max_tokens": 3,
        "temperature": 0,
        "logprobs": 5,
        "echo": False,
        "prompt_logprobs": None,
    }, name="completion_logprobs")

    choice = response.json["choices"][0]
    logprobs = choice["logprobs"]
    assert len(logprobs["tokens"]) == 3
    assert len(logprobs["token_logprobs"]) == 3
    assert len(logprobs["top_logprobs"]) == 3
    assert len(logprobs["text_offset"]) == 3
    assert all(len(row) <= 5 for row in logprobs["top_logprobs"])
    assert choice["text"] == "".join(logprobs["tokens"])


def test_streamed_text_completion_returns_requested_logprobs(client):
    stream = client.completion_stream({
        "prompt": "hello", "max_tokens": 3, "temperature": 0, "logprobs": 4,
    }, name="completion_stream_logprobs").expect_sse()
    payloads = [choice["logprobs"] for chunk in stream.chunks
                for choice in chunk.get("choices", []) if choice.get("logprobs")]
    assert len(payloads) == 1
    assert len(payloads[0]["tokens"]) == 3
    assert all(len(row) <= 4 for row in payloads[0]["top_logprobs"])
