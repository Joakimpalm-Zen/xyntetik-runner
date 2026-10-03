import importlib.util
import json
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "agent_torture", ROOT / "scripts" / "agent-torture.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_default_matrix_is_repeatable_and_balanced():
    """120: eight families divide evenly into it and the published bar is a
    >=100-request matrix. v2's 105/7 results (and v1's 100/5) stay valid on
    their own terms but are not case-for-case comparable with v3 — hence the
    SCHEMA_VERSION bump, which is what tells a reader which matrix a result
    file describes. v3 added tool_stream_normalization when the atem work
    taught the harness to verify streamed tool calls."""
    first = MOD.build_cases()
    second = MOD.build_cases()
    assert first == second
    assert len(first) == 120
    assert len({case["id"] for case in first}) == 120
    assert {case["category"] for case in first} == {
        "nested_arguments", "tool_selection", "forced_truncation",
        "stream_normalization", "tool_stream_normalization",
        "large_enum_selection", "reasoning_then_tool", "structured_final",
    }
    assert all(sum(c["category"] == category for c in first) == 15
               for category in {c["category"] for c in first})


def test_reasoning_then_tool_replays_prose_before_demanding_a_call():
    """The family exists to catch content bleeding into a call turn, so the
    request must actually carry an earlier assistant turn — a version that
    forgot it would still pass a 'one tool call' check while testing nothing."""
    cases = [c for c in MOD.build_cases() if c["category"] == "reasoning_then_tool"]
    assert cases
    messages = cases[0]["request"]["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "user"], roles
    assert len(messages[1]["content"]) > 40, "the replayed reasoning is trivial"
    assert cases[0]["request"]["tool_choice"]["function"]["name"] == "record_conclusion"


def test_structured_final_constrains_a_final_answer_not_a_call():
    """The other new family goes through response_format, which reaches the
    sampler by a different path than tools do."""
    cases = [c for c in MOD.build_cases() if c["category"] == "structured_final"]
    assert cases
    request = cases[0]["request"]
    assert "tools" not in request, "a structured final must not offer tools"
    schema = request["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["owner"]["required"] == ["team", "oncall"]
    # a valid document passes and a near-miss fails, so the verifier is real
    MOD.validate_against_schema(
        {"summary": "s", "severity": "high",
         "owner": {"team": "core", "oncall": True}}, MOD.FINAL_SCHEMA)
    try:
        MOD.validate_against_schema(
            {"summary": "s", "severity": "critical",
             "owner": {"team": "core", "oncall": True}}, MOD.FINAL_SCHEMA)
    except Exception:
        pass
    else:
        raise AssertionError("severity outside the enum was accepted")


def test_large_enum_case_constrains_to_an_exact_taxonomy_member():
    cases = [c for c in MOD.build_cases(100)
             if c["category"] == "large_enum_selection"]
    assert cases, "the large-enum family must be present in the matrix"
    case = cases[0]
    tool = case["request"]["tools"][0]["function"]
    assert tool["name"] == "classify_ticket"
    enum = tool["parameters"]["properties"]["label"]["enum"]
    assert len(enum) >= 50 and len(set(enum)) == len(enum)
    # a valid member passes, a plausible near-miss fails — the exact behaviour
    # the schema-constrained decoder must enforce on a small model
    MOD.validate_against_schema({"label": enum[0]}, MOD.CLASSIFY_SCHEMA)
    try:
        MOD.validate_against_schema({"label": "billing_issue"}, MOD.CLASSIFY_SCHEMA)
        assert False, "a non-member label must be rejected"
    except Exception:
        pass


def test_report_schema_and_totals(tmp_path):
    cases = MOD.build_cases(5)
    results = [
        MOD.result_for(cases[0], "passed", 1.25),
        MOD.result_for(cases[1], "failed", 2.5,
                       failure={"category": "schema", "message": "bad"}),
        MOD.result_for(cases[2], "passed", 3.0),
        MOD.result_for(cases[3], "failed", 4.0,
                       failure={"category": "protocol", "message": "bad"}),
        # a declined turn: no call at all. It is a failure, but on the OTHER
        # arm — "chose not to call", not "called wrongly".
        MOD.result_for(cases[4], "failed", 5.0,
                       failure={"category": "declined", "message": "prose"}),
    ]
    report = MOD.make_report(results, "runner", "runner test", "fixture.gguf",
                             123, 456)
    path = tmp_path / "report.json"
    MOD.write_json(path, report)
    decoded = json.loads(path.read_text())

    assert decoded["schema_version"] == "xyntetik.agent-torture.v4"
    assert decoded["runtime"] == {"name": "runner", "version": "runner test"}
    assert decoded["configuration"]["model"] == "fixture.gguf"
    assert decoded["totals"] == {
        "requests": 5, "excused": 0, "scored": 5, "passed": 2, "failed": 3,
        # both arms, reported separately: declined (chose not to call) split out
        # from the attempted cases, with the two rates.
        "declined": 1, "attempted": 4,
        "call_rate": 0.8, "attempted_pass_rate": 0.5,
        "failures_by_category": {"declined": 1, "protocol": 1, "schema": 1},
    }
    assert decoded["metrics"]["valid_structured_tasks_per_second"] == 16.26
    assert decoded["resources"]["peak_rss_kb"] == 456
    assert [c["id"] for c in decoded["cases"]] == [c["id"] for c in cases]


def test_report_is_labeled_with_the_runtime_under_test():
    # the whole point of cross-runtime: a report names which runtime produced
    # it, so llama.cpp / ollama / vllm results compare directly
    report = MOD.make_report([], "llama.cpp", "b3200", "qwen2.5-7b", 1000, None)
    assert report["runtime"] == {"name": "llama.cpp", "version": "b3200"}
    assert report["resources"]["peak_rss_kb"] is None  # foreign process


def test_spec_stats_are_aggregated_from_runner_log(tmp_path):
    log = tmp_path / "runner.log"
    log.write_text(
        "noise\n"
        "spec: 3 rounds, 12 drafted, 7 accepted (2.67 tok/round), grammar 2/3\n"
        "spec: 2 rounds, 4 drafted, 1 accepted (1.50 tok/round), grammar 0/0\n")
    assert MOD.read_spec_stats(log) == {
        "rounds": 5, "drafted": 16, "accepted": 8,
        "acceptance_rate": 0.5, "grammar_drafted": 3,
        "grammar_accepted": 2,
    }
    assert MOD.speculation_was_exercised(MOD.read_spec_stats(log))
    assert not MOD.speculation_was_exercised(MOD.read_spec_stats(
        tmp_path / "knob-blind.log"))


def test_runtime_axis_requires_case_for_case_identical_verdicts():
    cases = MOD.build_cases(2)
    plain = MOD.make_report(
        [MOD.result_for(cases[0], "passed", 1),
         MOD.result_for(cases[1], "failed", 1,
                        {"category": "schema", "message": "plain"})],
        "runner", "v", "target", 2, 3)
    draft = MOD.make_report(
        [MOD.result_for(cases[0], "passed", 1),
         MOD.result_for(cases[1], "failed", 1,
                        {"category": "schema", "message": "draft"})],
        "runner", "v", "target", 2, 3)
    assert MOD.compare_verdicts(plain, draft) == []

    # Negative control: a knob-blind harness that accidentally compares the
    # baseline report to itself cannot see this changed draft verdict.
    draft["cases"][1]["status"] = "passed"
    mismatches = MOD.compare_verdicts(plain, draft)
    assert mismatches == [{"id": cases[1]["id"],
                           "baseline": "failed", "draft": "passed"}]


def test_draft_server_flags_are_a_runtime_axis():
    assert MOD.runner_extra_args(None, 4) == ["--gpu", "off"]
    assert MOD.runner_extra_args(Path("small.gguf"), 6) == [
        "--gpu", "off", "--draft", "small.gguf", "--draft-k", "6"]
    with pytest.raises(ValueError, match="positive"):
        MOD.runner_extra_args(Path("small.gguf"), 0)


def test_endpoint_parsing_accepts_host_port_and_urls_rejects_remote():
    assert MOD._parse_endpoint("127.0.0.1:8080") == 8080
    assert MOD._parse_endpoint("http://localhost:11434/") == 11434
    assert MOD._parse_endpoint("localhost:8000") == 8000
    for bad in ("127.0.0.1", "10.0.0.5:8080", "example.com:8080"):
        try:
            MOD._parse_endpoint(bad)
            assert False, f"{bad} should be rejected"
        except ValueError:
            pass


def test_remote_target_exposes_the_client_contract():
    # the harness Client drives a target through .port / assert_alive /
    # sample_rss — a RemoteTarget provides exactly that, so the same matrix
    # runs against any local OpenAI-compatible server
    target = MOD.RemoteTarget(65535)
    assert target.port == 65535
    assert target.peak_rss_kb is None
    assert target.sample_rss() is None


def test_stream_normalization_is_independent_of_tcp_chunks():
    raw = (b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
           b'data: {"choices":[{"delta":{"content":"b"},'
           b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    reference = MOD.normalize_sse(raw, [raw])
    assert reference["text"] == "ab"
    assert reference["finish_reason"] == "stop"
    assert reference["saw_done"] is True
    for point in range(len(raw) + 1):
        assert MOD.normalize_sse(raw, [raw[:point], raw[point:]]) == reference


def _reasoning_limit_no_call(budget):
    case = next(c for c in MOD.build_cases(40) if c["category"] == "forced_truncation")
    case = dict(case, request=dict(case["request"], max_tokens=budget))

    class Resp:
        status = 200
        choice = {"message": {"content": "", "reasoning_content": "\n"},
                  "finish_reason": "length"}
        json = {"choices": [choice],
                "runner_telemetry": {"finish_detail": "reasoning_limit"}}

        def expect_status(self, code):
            assert code == 200
            return self

    return case, Resp()


def test_only_the_one_token_budget_is_excused():
    """R4.12.3, narrowed: under a constraint the engine closes reasoning at
    half the budget (prelude_max), so only max_tokens 1 cannot be split. A
    reasoning_limit no-call there is excused; at 2, 3, 5 or 8 the same
    response means the reserve regressed, and it fails and is scored."""
    import pytest as _pt
    case, resp = _reasoning_limit_no_call(1)
    with _pt.raises(MOD.DegenerateBudget):
        MOD._verify_buffered(case, resp)
    for budget in (2, 3, 5, 8):
        case, resp = _reasoning_limit_no_call(budget)
        with _pt.raises(MOD.ProtocolError) as err:
            MOD._verify_buffered(case, resp)
        assert not isinstance(err.value, (MOD.DegenerateBudget, MOD.Declined)), budget
        assert "reserve did not hold" in str(err.value)
    excused, _ = _reasoning_limit_no_call(1)
    failed, _ = _reasoning_limit_no_call(2)
    results = [
        MOD.result_for(excused, "excused", 1.0,
                       failure={"category": "degenerate_budget", "message": "m"}),
        MOD.result_for(failed, "failed", 1.0,
                       failure={"category": "protocol", "message": "reserve"}),
    ]
    t = MOD.make_report(results, "runner", "v", "m.gguf", 10, 1)["totals"]
    assert t["requests"] == 2 and t["excused"] == 1 and t["scored"] == 1
    assert t["failed"] == 1 and t["passed"] == 0


def test_a_budget_spent_in_reasoning_is_excused_not_failed():
    """R4.12.3 (owner 2026-10-03): a forced-truncation case whose budget ended
    inside the reasoning channel (finish_detail reasoning_limit, no call) is
    excused and leaves the denominators; one that simply declined still counts
    against the call rate."""
    cases = [c for c in MOD.build_cases(40) if c["category"] == "forced_truncation"][:3]
    # the excused budget is the unsplittable one (see the narrowed rule)
    cases[0] = dict(cases[0], request=dict(cases[0]["request"], max_tokens=1))

    class Resp:
        def __init__(self, detail):
            self.status = 200
            self.choice = {"message": {"content": "", "reasoning_content": ""},
                           "finish_reason": "length"}
            self.json = {"choices": [self.choice],
                         "runner_telemetry": {"finish_detail": detail} if detail else {}}

        def expect_status(self, code):
            assert code == 200
            return self

    import pytest as _pt
    with _pt.raises(MOD.DegenerateBudget):
        MOD._verify_buffered(cases[0], Resp("reasoning_limit"))
    with _pt.raises(MOD.Declined):
        MOD._verify_buffered(cases[1], Resp(None))
    results = [
        MOD.result_for(cases[0], "excused", 1.0,
                       failure={"category": "degenerate_budget", "message": "m"}),
        MOD.result_for(cases[1], "failed", 1.0,
                       failure={"category": "declined", "message": "prose"}),
        MOD.result_for(cases[2], "passed", 1.0),
    ]
    t = MOD.make_report(results, "runner", "v", "m.gguf", 10, 1)["totals"]
    assert t["requests"] == 3 and t["excused"] == 1 and t["scored"] == 2
    assert t["passed"] == 1 and t["failed"] == 1 and t["declined"] == 1
    assert t["attempted"] == 1 and t["attempted_pass_rate"] == 1.0
