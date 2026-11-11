"""R6.51 TEMPORARY: one run's numbers from the probe's files, as log lines."""
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
run = sys.argv[2] if len(sys.argv) > 2 else "?"


def lines(name):
    p = out / name
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


env = (lines("env.json") or [{}])[0]
samples = lines("samples.jsonl")
tests = lines("tests.jsonl")
target = lines("target.jsonl")
snaps = lines("snapshots.jsonl")
final = (lines("final.json") or [{}])[0]

print(f"R6.51 env: dynamic range {env.get('dynamic_range')}, tcpip parameters {env.get('tcpip_parameters')}")
print(f"R6.51 env ipv6: {' '.join(str(env.get('netsh_ipv6', '')).split())}")
print(f"R6.51 harness: {final.get('connects')} connects from the pytest process over {final.get('t')} s, "
      f"{len([t for t in tests if t['when'] == 'call'])} tests called")

calls = [t for t in tests if t["when"] == "call"]
for t in sorted(calls, key=lambda t: -t["connects"])[:8]:
    print(f"R6.51 top connects: {t['connects']:6d} {t['outcome']:7s} {t['seconds']:7.1f}s {t['test']}")

for e in target:
    tab = e["table"]
    print(f"R6.51 target {e['event']}: t={e['t']}s connects so far {e['connects']}, last 120 s "
          f"{e['connects_last_120s']}; dynamic ports held {tab['dynamic_ports_held']}/{tab['dynamic_range_size']}, "
          f"TIME_WAIT client {tab['time_wait_client']} server {tab['time_wait_server']}, states {tab['states']}")

if samples:
    peak = max(samples, key=lambda s: s["table"]["dynamic_ports_held"])
    tab = peak["table"]
    print(f"R6.51 peak: dynamic ports held {tab['dynamic_ports_held']}/{tab['dynamic_range_size']} at t={peak['t']}s "
          f"in {peak['test']}; TIME_WAIT client {tab['time_wait_client']} server {tab['time_wait_server']}; "
          f"connects last 120 s {peak['connects_last_120s']}")
    tw = max(samples, key=lambda s: s["table"]["time_wait_client"] + s["table"]["time_wait_server"])
    print(f"R6.51 peak TIME_WAIT: client {tw['table']['time_wait_client']} server {tw['table']['time_wait_server']} "
          f"at t={tw['t']}s in {tw['test']}")
    np_ = [s["nonpaged_bytes"] for s in samples if s.get("nonpaged_bytes")]
    if np_:
        print(f"R6.51 nonpaged pool: min {min(np_) / 2**20:.1f} MiB, max {max(np_) / 2**20:.1f} MiB "
              f"over {len(np_)} readings")
    in_target = [s for s in samples if s.get("test") and "test_list_contexts_during_creation" in s["test"]]
    if in_target:
        p = max(in_target, key=lambda s: s["table"]["dynamic_ports_held"])
        print(f"R6.51 peak in target: dynamic ports held {p['table']['dynamic_ports_held']}, "
              f"TIME_WAIT client {p['table']['time_wait_client']} server {p['table']['time_wait_server']}")

for s in snaps:
    print(f"R6.51 SNAPSHOT at {s['why']}: t={s['t']}s in {s['test']}; connects {s['connects']}, last 120 s "
          f"{s['connects_last_120s']}; table {s['table']}; nonpaged {s.get('nonpaged_bytes')}")

outcome = next((t["outcome"] for t in calls if "test_list_contexts_during_creation" in t["test"]), "not run")
failed = [t["test"] for t in tests if t["outcome"] == "failed"]
held = max((s["table"]["dynamic_ports_held"] for s in samples), default=None)
print(f"R6.51 RESULT run={run} target={outcome} reproduced={'yes' if snaps else 'no'} "
      f"failed={len(failed)} peak_dynamic_ports_held={held} range={env.get('dynamic_range')}")
for f in failed:
    print(f"R6.51 failed: {f}")
