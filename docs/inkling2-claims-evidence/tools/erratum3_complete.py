#!/usr/bin/env python3
"""R4.48.12 ERRATUM 3 (2026-11-24): Runner's complete Claim 1 generations, read from each committed record's own fields.

The committed Runner records (records/claim1-runner-ref/answers.jsonl, written by claim_client.py version 1) store the
receipt's output ids as gen_ids, and Runner's receipt lists the ids WITHOUT the stop token, while the same record's
response reports it (choices[0].stop_token_id). This script writes a DERIVED file in which each record's gen_ids is
the complete generation, using claim_client_v2.py's own with_stop() (committed b98a60442): the receipt's ids plus
[stop_token_id] when the response reports a stop token (an int) and the ids do not already end with it. Every other
field is copied unchanged; each record also carries gen_ids_receipt, stop_token_appended and stop_token_id. The
committed records are read, never written.

usage: erratum3_complete.py CLIENT_V2.py ANSWERS.jsonl OUT.jsonl
"""
import importlib.util, json, sys


def main():
    client, src, dst = sys.argv[1:4]
    spec = importlib.util.spec_from_file_location("claim_client_v2", client)
    c2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c2)
    n = appended = 0
    with open(dst, "w") as fo:
        for line in open(src):
            r = json.loads(line)
            gen, app, st = c2.with_stop(r["gen_ids"], r["response"])
            out = dict(r)
            out.update({"gen_ids": gen, "gen_ids_receipt": r["gen_ids"], "stop_token_appended": app,
                        "stop_token_id": st, "erratum": 3})
            fo.write(json.dumps(out) + "\n")
            n += 1
            appended += app
    print("records %d, stop token appended %d, unchanged %d" % (n, appended, n - appended))


if __name__ == "__main__":
    main()
