#!/usr/bin/env python3
"""R4.29 row 3: runner main's scripts/template-conformance.py with one lab-local family added, Qwen3.6-27B's own
template as the oracle (Qwen/Qwen3.6-27B's chat_template.jinja, byte-identical to the qwen36-35b-a3b family's),
rendered by Runner's qwen36 renderer (template_detect's pick for it). The runner checkout is not modified; the family exists only in this process.

    conformance.py RUNNER_DIR [template-conformance.py args ...]
"""
import importlib.util
import os
import sys

runner = sys.argv[1]
sys.path.insert(0, os.path.join(runner, "scripts"))   # its helpers (gguf_fixture, ...) sit beside it
spec = importlib.util.spec_from_file_location("template_conformance",
                                              os.path.join(runner, "scripts", "template-conformance.py"))
tc = importlib.util.module_from_spec(spec)
sys.modules["template_conformance"] = tc
spec.loader.exec_module(tc)
tc.FAMILIES["qwen36-27b"] = tc.Family(
    "qwen36", ("hf", "Qwen/Qwen3.6-27B"),
    note="R4.29 lab check: Qwen 3.6's template (preserve_thinking; non-string tool arguments through tojson) "
         "against the qwen36 renderer",
    tool_family=True, hostile_args=True, thinking_var="enable_thinking",
    tokenizer=("Qwen3.6-27B-Q4_K_M.gguf",),
    cannot={"system-mid-history": tc.NO_MID_SYSTEM})
sys.argv = [os.path.join(runner, "scripts", "template-conformance.py")] + sys.argv[2:]
sys.exit(tc.main())
