"""Per-process memory and cumulative throughput, seen from the wire.

These are the two runner-side measurements a capacity dashboard cannot get any
other way. A supervisor watching several runners needs to know what each one is
actually costing in RAM, and how much work it has done, and neither is
answerable from outside the process: RSS attributed to a PID is not the same as
the model's file size (weights are shared, mapped pages come and go), and
tokens-per-second computed by timing HTTP requests measures the client's
network as much as the server's decode.

The boundary this respects: the runner MEASURES and reports raw counters, it
does not PRESENT rates. `tokens_generated` and `generate_seconds` are exposed
as monotonic totals so a dashboard can difference them over whatever window it
cares about. A tok/s field would bake in an averaging window the runner has no
business choosing, and would be wrong for every consumer whose window differs.
"""

import json


def health(client):
    r = client.get("/health", name="health")
    assert r.status == 200, r.status
    return json.loads(r.body)


def test_health_reports_process_memory(client):
    """RSS for THIS process, which is the number a supervisor budgets against."""
    h = health(client)
    assert "rss_bytes" in h, h
    assert "peak_rss_bytes" in h, h
    # A loaded server is worth more than a few pages; a wildly large value
    # means the units are wrong (kilobytes reported as bytes, say).
    assert h["rss_bytes"] > (1 << 20), h["rss_bytes"]
    assert h["rss_bytes"] < (1 << 42), h["rss_bytes"]
    assert h["peak_rss_bytes"] >= h["rss_bytes"], h


def test_throughput_counters_are_monotonic_and_start_sane(client):
    h = health(client)
    for key in ("tokens_generated", "tokens_prompt", "generate_seconds"):
        assert key in h, (key, h)
        assert h[key] >= 0, (key, h[key])


def test_generating_advances_the_counters(client):
    """The counters must count the work, not merely exist."""
    before = health(client)
    body = {"model": "test",
            "messages": [{"role": "user", "content": "count to three"}],
            "temperature": 0, "max_tokens": 8}
    r = client.chat(body, name="metrics-chat").expect_status(200)
    used = r.usage
    after = health(client)

    gained = after["tokens_generated"] - before["tokens_generated"]
    assert gained == used["completion_tokens"], (gained, used)
    assert after["tokens_prompt"] - before["tokens_prompt"] == used["prompt_tokens"]
    # Decode took real time, and the counter is cumulative so it only grows.
    assert after["generate_seconds"] > before["generate_seconds"], (
        before["generate_seconds"], after["generate_seconds"])


def test_health_does_not_count_itself(client):
    """Polling /health must not move the work counters.

    A dashboard polls this endpoint on a timer. If the poll registered as
    activity the dashboard would be measuring its own polling, which is the
    same trap `active_requests` already documents for itself.
    """
    a = health(client)
    for _ in range(3):
        health(client)
    b = health(client)
    assert b["tokens_generated"] == a["tokens_generated"]
    assert b["tokens_prompt"] == a["tokens_prompt"]
    assert b["generate_seconds"] == a["generate_seconds"]


def test_batch_counters_report_the_work_the_scheduler_did(client):
    """How much batching actually happened, in the same raw-counter form.

    The scheduler counts microbatch steps and the sequences cut into them. That
    pair is the only wire-visible answer to "is continuous batching earning its
    thread on this box": `batch_sequences / batch_steps` is the mean batch size
    over any window a dashboard cares to difference, and a single-slot or
    non-batched server reports a flat zero rather than a misleading one.

    Kept raw for the same reason `generate_seconds` is: the ratio needs a
    window, and the runner has no business choosing one.
    """
    before = health(client)
    for key in ("batch_steps", "batch_sequences"):
        assert key in before, (key, before)
        assert before[key] >= 0, (key, before[key])

    body = {"model": "test",
            "messages": [{"role": "user", "content": "count to three"}],
            "temperature": 0, "max_tokens": 8, "cache_prompt": False}
    client.chat(body, name="batch-metrics-chat").expect_status(200)
    after = health(client)

    # this suite's server runs two slots, so batching is on (the banner says
    # "continuous batching") and a generation must move the step counter
    assert after["batch_steps"] > before["batch_steps"], (before, after)
    # every step batches at least the one sequence that woke it
    assert after["batch_sequences"] >= after["batch_steps"], after


# --------------------------------------------------------------------- /metrics
#
# The same facts in Prometheus text exposition 0.0.4, because that is what a
# monitoring stack ingests without a translator in front of it. The parsing
# here is deliberately strict and hand-written rather than borrowed from a
# client library: the point of these tests is that a stock scraper can read
# the body, and a lenient parser of our own would hide exactly the mistakes
# that would break one.

METRICS_CONTENT_TYPE = "text/plain; version=0.0.4"

# Prometheus' own naming rules, restated here rather than derived from the
# server's output: a counter's name ends in `_total`, a gauge's does not, and
# every name in this exposition carries the `runner_` prefix. Deriving the
# expectation from the response would make this test agree with whatever the
# server said.
EXPECTED_COUNTERS = {
    "runner_requests_total",
    "runner_prompt_tokens_total",
    "runner_prompt_cached_tokens_total",
    "runner_generated_tokens_total",
    "runner_generate_seconds_total",
    "runner_batch_steps_total",
    "runner_batch_sequences_total",
    "runner_prefix_cache_hits_total",
    "runner_prefix_cache_misses_total",
    "runner_prefix_cache_stores_total",
    "runner_prefix_cache_evictions_total",
    "runner_prefix_cache_tokens_reused_total",
    "runner_prefix_cache_saved_prefill_seconds_total",
    "runner_speculation_rounds_total",
    "runner_speculation_drafted_tokens_total",
    "runner_speculation_accepted_tokens_total",
}
EXPECTED_GAUGES = {
    "runner_active_requests",
    "runner_resident_memory_bytes",
    "runner_peak_resident_memory_bytes",
    "runner_prefix_cache_bytes",
    "runner_prefix_cache_budget_bytes",
    "runner_prefix_cache_entries",
}


def parse_exposition(text):
    """Parse Prometheus text exposition, refusing anything a scraper would.

    Returns {name: (type, value)}. Every sample must be preceded by its own
    HELP and TYPE lines, each name may appear once, and nothing else is
    allowed on a line.
    """
    helps, types, samples = {}, {}, {}
    for lineno, line in enumerate(text.split("\n"), 1):
        if line == "":
            continue
        if line.startswith("# HELP "):
            name, _, doc = line[len("# HELP "):].partition(" ")
            assert name not in helps, (lineno, "duplicate HELP", name)
            assert doc, (lineno, "HELP with no text", name)
            helps[name] = doc
            continue
        if line.startswith("# TYPE "):
            name, _, kind = line[len("# TYPE "):].partition(" ")
            assert name not in types, (lineno, "duplicate TYPE", name)
            assert kind in ("counter", "gauge"), (lineno, kind)
            types[name] = kind
            continue
        assert not line.startswith("#"), (lineno, "unknown comment line", line)
        parts = line.split(" ")
        assert len(parts) == 2, (lineno, "not one sample per line", line)
        name, raw = parts
        assert name in helps, (lineno, "sample before its HELP", name)
        assert name in types, (lineno, "sample before its TYPE", name)
        assert name not in samples, (lineno, "duplicate sample", name)
        samples[name] = (types[name], float(raw))
    assert set(helps) == set(samples), (sorted(set(helps) ^ set(samples)),)
    assert set(types) == set(samples), (sorted(set(types) ^ set(samples)),)
    return samples


def metrics(client):
    r = client.get("/metrics", name="metrics")
    assert r.status == 200, r.status
    ctype = r.headers.get("content-type")
    assert ctype == METRICS_CONTENT_TYPE, ctype
    declared = r.headers.get("content-length")
    assert declared is not None and int(declared) == len(r.body), (
        declared, len(r.body))
    return parse_exposition(r.body.decode("utf-8"))


def test_metrics_exposition_parses_and_names_are_prometheus_shaped(client):
    m = metrics(client)
    missing = (EXPECTED_COUNTERS | EXPECTED_GAUGES) - set(m)
    assert not missing, sorted(missing)
    for name, (kind, value) in m.items():
        assert name.startswith("runner_"), name
        assert value >= 0, (name, value)
        if kind == "counter":
            assert name.endswith("_total"), f"{name} is a counter but is not _total"
            assert name in EXPECTED_COUNTERS, f"{name} is an undeclared counter"
        else:
            assert not name.endswith("_total"), f"{name} is a gauge named _total"
            assert name in EXPECTED_GAUGES, f"{name} is an undeclared gauge"


def test_metrics_counters_advance_with_the_work(client):
    """Counters must count, and must only ever go up.

    Two generations, so the delta is a fact about this test rather than about
    whatever the suite ran before it."""
    before = metrics(client)
    body = {"model": "test",
            "messages": [{"role": "user", "content": "count to three"}],
            "temperature": 0, "max_tokens": 8}
    a = client.chat(body, name="metrics-prom-1").expect_status(200)
    b = client.chat(body, name="metrics-prom-2").expect_status(200)
    after = metrics(client)

    for name in EXPECTED_COUNTERS:
        assert after[name][1] >= before[name][1], (
            f"{name} went backwards", before[name], after[name])

    used = [a.usage, b.usage]
    assert after["runner_requests_total"][1] - \
        before["runner_requests_total"][1] == 2, (before, after)
    assert after["runner_generated_tokens_total"][1] - \
        before["runner_generated_tokens_total"][1] == \
        sum(u["completion_tokens"] for u in used)
    assert after["runner_prompt_tokens_total"][1] - \
        before["runner_prompt_tokens_total"][1] == \
        sum(u["prompt_tokens"] for u in used)
    # The counter and the per-request wire field are the same measurement,
    # which is the whole reason both exist: a dashboard differencing the
    # counter and a client reading its own usage must not disagree.
    assert after["runner_prompt_cached_tokens_total"][1] - \
        before["runner_prompt_cached_tokens_total"][1] == \
        sum(u["prompt_tokens_details"]["cached_tokens"] for u in used)


def test_metrics_agrees_with_health_and_the_prefix_cache_route(client):
    """Three routes, one set of counters.

    The independent anchor for this endpoint: /health and
    /v1/runner/prefix-cache have their own tests and their own callers, so a
    /metrics that disagrees with them is wrong no matter how well it parses.
    Read /metrics between two identical reads of the others, so ordinary
    concurrent traffic cannot make an equality fail.
    """
    h_before = health(client)
    pfx_before = json.loads(client.get("/v1/runner/prefix-cache",
                                       name="metrics-pfx-before").body)
    m = metrics(client)
    h_after = health(client)
    pfx_after = json.loads(client.get("/v1/runner/prefix-cache",
                                      name="metrics-pfx-after").body)

    pairs = [("runner_prompt_tokens_total", "tokens_prompt", h_before, h_after),
             ("runner_generated_tokens_total", "tokens_generated",
              h_before, h_after),
             ("runner_batch_steps_total", "batch_steps", h_before, h_after),
             ("runner_batch_sequences_total", "batch_sequences",
              h_before, h_after),
             ("runner_prefix_cache_hits_total", "hits", pfx_before, pfx_after),
             ("runner_prefix_cache_misses_total", "misses",
              pfx_before, pfx_after),
             ("runner_prefix_cache_bytes", "bytes", pfx_before, pfx_after),
             ("runner_prefix_cache_entries", "entries", pfx_before, pfx_after)]
    for metric, field, before, after in pairs:
        # sorted, not (before, after): two of these are gauges, and a gauge is
        # allowed to fall (an eviction) between the two reads.
        low, high = sorted((before[field], after[field]))
        assert low <= m[metric][1] <= high, (
            metric, field, before[field], m[metric][1], after[field])


def test_metrics_does_not_count_itself(client):
    """Scraping is not work. A scraper polls this on a timer, and a request
    counter that included the scrape would report the monitoring instead of
    the model."""
    a = metrics(client)
    for _ in range(3):
        metrics(client)
    b = metrics(client)
    for name in ("runner_requests_total", "runner_prompt_tokens_total",
                 "runner_generated_tokens_total"):
        assert a[name] == b[name], (name, a[name], b[name])


def test_metrics_is_advertised_in_capabilities(client):
    """A scraper's alternative is polling /health and translating, so it has
    to be able to tell which server it is pointed at without guessing."""
    caps = json.loads(client.get("/v1/capabilities", name="metrics-caps").body)
    assert caps["features"]["prometheus_metrics"] is True, caps["features"]
