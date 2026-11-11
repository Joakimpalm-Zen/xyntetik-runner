#!/usr/bin/env python3
"""Every public surface of Runner and the one component that owns it (R18.2).

`docs/ownership.md` names seven components. This script reads the surfaces out
of the source, gives each one an owner by the rules below, and writes
`docs/capability-inventory.md`. The surfaces are:

- the command line: every option `runner --help` declares, and the
  subcommands with the `runner attest` verbs;
- every HTTP route `server.c` matches;
- every `RUNNER_*` switch read through `getenv` or an environment helper;
- every record format (`xyntetik.runner.*.vN`);
- the Python client's `__all__` and the TypeScript client's exports;
- every file under `src/`.

There is no default owner. A surface no rule claims is an error, so a new
flag, route, switch, record or client name fails the check until someone
says whose it is. A rule that claims nothing is an error too, so the rules
cannot outlive what they were written for.

Usage:
    python3 scripts/capability-inventory.py --write   # regenerate the page
    python3 scripts/capability-inventory.py --check   # the page is current and nothing is unowned
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "capability-inventory.md"

COMPONENTS = ["runtime", "execution", "substrate", "production", "proof", "shadow", "desktop"]
TITLES = {
    "runtime": "Runtime", "execution": "Execution", "substrate": "Model substrate",
    "production": "Model production", "proof": "Proof", "shadow": "Shadow",
    "desktop": "Desktop", "shared": "Shared",
}

# ---------------------------------------------------------------- the rules

# Source files by stem (`model` covers model.c and model.h). `shared` is the
# three files ownership.md names as carrying more than one owner, plus the
# umbrella header; a switch read only there needs a name rule below.
SOURCE_RULES = [
    ("execution", re.compile(r"^kernels")),
    ("execution", {"cuda", "metal", "gpu", "gpu_none", "engine", "scheduler", "quants",
                   "quants_iq_grids", "tpool", "act", "dn_eps", "fp16", "fpclass", "fpenv",
                   "iq_decode", "kernel_args", "kq_stage", "metal_admission", "pmath",
                   "upload_report"}),
    ("runtime", {"server", "server_int", "http", "api", "api_anthropic", "api_responses",
                 "schema", "schema_int", "schema_tools", "jsonmode", "sample", "template",
                 "registry", "session", "respstore", "kvsnap", "decide", "clef", "vramreg",
                 "hfhub", "model_roster"}),
    ("substrate", {"gguf", "tokenizer", "unicode_classes", "unicode_nfc", "npy", "json",
                   "compat"}),
    ("production", {"quantize", "derive", "dpo", "tdn_args"}),
    ("proof", {"receipts", "bundle", "provenance", "envelope", "lineage", "oms", "ecdsa",
               "ed25519", "mldsa", "watermark", "build_arch"}),
    ("desktop", {"tray", "tray_win", "tray_macos", "tray_stub", "instances"}),
    ("shared", {"main", "model", "completion", "runner"}),
]

CLI_RULES = {
    "runtime": {
        "-p", "-f", "-i", "-n", "-s", "-hf", "-h", "-v", "--help", "--version", "--serve",
        "--port", "--parallel", "--ttl", "--info", "--json", "--pull", "--system", "--temp",
        "--top-k", "--top-p", "--min-p", "--repeat-penalty", "--think", "--reasoning-budget",
        "--reasoning-budget-message", "--reasoning-temp", "--loop-guard", "--ignore-eos",
        "--chat-template", "--bare-recipients", "--decide", "--systemone", "--score",
        "--score-out", "--score-tokens", "--sessions", "--session-out", "--resume",
        "--fork-seed", "--suspend-after", "--kv-snapshots", "--reserve", "--wait-for-vram",
        "--vram-priority", "--yield-on-request", "--parent-pid", "--memory", "--ram-share",
        "--roster", "--sizes-only", "--doctor-include-text",
        "runner recommend", "runner cached",
    },
    "execution": {
        "-t", "-b", "-c", "--gpu", "--gpu-route", "--kv", "--kv-ring", "--rope", "--cpu-moe",
        "--moe-prefetch", "--mlock", "--draft", "--draft-k", "--draft-required",
        "--bench-json",
    },
    "substrate": {"-m", "--lora", "--lora-scale", "--adapter", "--no-bos"},
    "production": {
        "--rewrite", "--quant", "--type-plan", "--type-plan-strict", "--imatrix",
        "--prune-experts", "--remove-sublayer", "--merge-lora", "--merge-allow-erased",
        "--context-surgery", "--quant-error", "runner train", "runner derive",
    },
    "proof": {
        "--record", "--record-keep", "--record-prev", "--verify", "--sign-key", "--trust-key",
        "--require-signed", "--require-signed-model", "--model-sig", "--model-pubkey",
        "--lora-sig", "--require-eval", "--watermark", "--force-uncertified",
        "runner attest", "runner attest keygen", "runner attest sign", "runner attest check",
        "runner attest lineage", "runner attest bundle", "runner attest check-bundle",
        "runner attest pack", "runner attest check-pack", "runner attest sign-model",
        "runner attest watermark-keygen", "runner attest detect",
    },
    "shadow": {"--shadow-mode"},
    "desktop": {"--tray"},
}

HTTP_RULES = {
    "runtime": {
        "/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/responses/{id}",
        "/v1/messages", "/v1/messages/count_tokens", "/v1/embeddings", "/v1/rerank",
        "/v1/decide", "/v1/systemone", "/v1/models", "/v1/capabilities", "/health",
        "/metrics", "/unload", "/v1/runner/contexts", "/v1/runner/contexts/{id}",
        "/v1/runner/sessions", "/v1/runner/sessions/{id}", "/v1/runner/prefix-cache",
        "/v1/runner/prefix-cache/clear", "/v1/runner/requests/{id}",
    },
    "proof": {"/v1/runner/provenance"},
}

# Switches: the first name rule that matches decides. Otherwise the owner is
# the component of the file that reads the switch, if every file that reads it
# has the same one. A switch read in a shared file, or in files of two
# components, needs a name rule.
ENV_NAME_RULES = [
    ("production", re.compile(r"^(TRAIN_|LORA_ADAM_|GOLD_)")),
    ("runtime", re.compile(r"^(SCHEMA_|PREFIX_CACHE_|SCORE_|TEST_SCRIPTED_REPLY$|"
                           r"TEST_STREAM_OUT_CAP$|VRAM_PRIORITY$)")),
    ("substrate", re.compile(r"^(ALLOW_UNKNOWN_ARCH|PREFETCH|TEST_LOAD_DELAY_MS|"
                             r"TEST_NO_FILE_ID|LORA_INSTALL_FAIL)$")),
    ("execution", re.compile(r"^(ACT_TRACE|ATTN_|CPU_MOE_DEFAULT$|DEBUG_(ACT|ROUTE|TIME|VRAM)$|GPU_FALLBACK$|"
                             r"KV_|LAYER_SIM$|MLA_|MOE_|MTP_|MV_FUSE$|SPEC_BATCH$|TIEDV|"
                             r"TEST_VRAM_FREE_MB$)")),
]

RECORD_RULES = {
    "runtime": {"session", "kv_snapshot", "cached", "decision", "doctor", "model-tiers",
                "score"},
    "production": {"train", "merge", "quant", "derive", "eval", "context-surgery"},
    "proof": {"transcript", "verify", "bundle", "envelope", "evidence-pack", "signkey",
              "watermark_key", "watermark.key_id", "watermark_detect"},
}

PYTHON_RULES = {"runtime": {"xyntetik_runner"}, "shadow": {"xyntetik_runner.shadow"}}

TYPESCRIPT_RULES = {
    "runtime": {"RunnerClient", "RunnerClientOptions", "RunnerAPIError", "RunnerProtocolError",
                "runnerTelemetry", "finishDetail", "choiceLogprobs", "SamplingTelemetry",
                "ToolProtocolTelemetry", "TimingTelemetry", "RepeatedToolCall",
                "SpeculationTelemetry", "RunnerTelemetry", "ChoiceLogprob", "RerankResult",
                "RerankResponse", "NamedContext", "RunnerErrorBody"},
    "proof": {"verifyTranscriptChain", "ChainCheck", "TranscriptRecord", "ReceiptFile",
              "Provenance", "ModelSignature", "EnvelopeState"},
}

# ---------------------------------------------------------------- reading the source

SRC_SUFFIXES = {".c", ".h", ".m", ".cu", ".cuh", ".metal"}
GENERATED = re.compile(r"^kernels_(metal|ptx|ptx80|tensor_metal)\.h$")
ENV_READ = re.compile(r'\b\w*(?:getenv|env)\w*\(\s*(?:&?\w+\s*,\s*)?"RUNNER_([A-Z0-9_]+)"')
RECORD_ID = re.compile(r"xyntetik\.runner\.([a-z0-9_.-]+)\.v(\d+)")
C_STRING = re.compile(r'"((?:[^"\\\n]|\\.)*)"')


def _text(path):
    return path.read_text(encoding="utf-8", errors="replace")


def _sources(root):
    return sorted(p for p in (root / "src").iterdir()
                  if p.suffix in SRC_SUFFIXES and not GENERATED.match(p.name))


def _function(src, head):
    i = src.index(head)
    return src[i:src.index("\n}\n", i)]


def _c_text(body):
    """The concatenated string literals of a C body, unescaped."""
    out = []
    for m in C_STRING.finditer(body):
        out.append(m.group(1).replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\"))
    return "".join(out)


def cli_surfaces(root):
    """The options `runner --help` declares, the subcommands and the attest verbs.

    The option rule is help-parity's: a line of the help text that opens with
    two spaces and a dash declares every long option on it, and its leading
    short option."""
    main = _text(root / "src" / "main.c")
    usage = _c_text(_function(main, "static void usage_to(FILE *f, const char *prog) {"))
    found = set()
    for line in usage.splitlines():
        if re.match(r"  -", line):
            found.update(re.findall(r"--[a-z][a-z0-9-]+", line))
            short = re.match(r"  (-[A-Za-z]+)\b", line)
            if short:
                found.add(short.group(1))
        sub = re.match(r"\s*(?:usage: %s .*|runner (\w+))", line)
        if sub and sub.group(1):
            found.add("runner " + sub.group(1))
    i = main.index("ATTEST_VERBS[] =")
    verbs = _c_text(main[i:main.index(";\n", i)])
    for line in verbs.splitlines():
        verb = re.match(r"  ([a-z][a-z-]*)\b", line)
        if verb:
            found.add("runner attest " + verb.group(1))
    if len(found) < 50:
        raise SystemExit(f"capability-inventory: read only {len(found)} command-line surfaces; "
                         "the help text's format changed")
    return {name: "src/main.c" for name in found}


def http_surfaces(root):
    """Every path `server.c` compares a request against. A prefix ending in
    `/` is a route with a trailing id."""
    src = _text(root / "src" / "server.c")
    lits = re.findall(r'\bstrn?cmp\(\s*path\s*,\s*"(/[^"]*)"', src)
    lits += re.findall(r'#define\s+\w+\s+"(/v1/[^"]*)"', src)
    found = {}
    for p in lits:
        found[p + "{id}" if p.endswith("/") else p] = "src/server.c"
    if len(found) < 10:
        raise SystemExit("capability-inventory: read fewer than 10 HTTP routes out of server.c")
    return found


def env_surfaces(root):
    found = {}
    for p in _sources(root):
        for name in ENV_READ.findall(_text(p)):
            found.setdefault(name, set()).add(p.relative_to(root).as_posix())
    return found


def record_surfaces(root):
    found = {}
    dirs = [root / "src", root / "python" / "src", root / "clients" / "typescript" / "src"]
    for d in dirs:
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix in SRC_SUFFIXES | {".py", ".ts"} \
                    and not GENERATED.match(p.name):
                for stem, ver in RECORD_ID.findall(_text(p)):
                    found.setdefault(f"{stem}.v{ver}", set()).add(p.relative_to(root).as_posix())
    return found


def python_surfaces(root):
    found = {}
    base = root / "python" / "src"
    for init in sorted(base.rglob("__init__.py")):
        module = ".".join(init.parent.relative_to(base).parts)
        m = re.search(r"^__all__ = \[(.*?)\]", _text(init), re.S | re.M)
        if not m:
            raise SystemExit(f"capability-inventory: {init.relative_to(root)} has no __all__")
        for name in re.findall(r'"([A-Za-z_]\w*)"', m.group(1)):
            found[f"{module}.{name}"] = module
    return found


def typescript_surfaces(root):
    found = {}
    for p in sorted((root / "clients" / "typescript" / "src").glob("*.ts")):
        for name in re.findall(r"^export (?:async )?(?:function|class|interface|type|const|enum) "
                               r"(\w+)", _text(p), re.M):
            found[name] = p.relative_to(root).as_posix()
    return found


def source_surfaces(root):
    return {p.relative_to(root).as_posix(): p.stem for p in _sources(root)}

# ---------------------------------------------------------------- owners


class Inventory:
    def __init__(self):
        self.owner = {}     # kind -> {surface: component}
        self.errors = []

    def claim(self, kind, surfaces, owner_of):
        self.owner[kind] = {}
        for name in sorted(surfaces):
            got = owner_of(name)
            if got is None:
                self.errors.append(f"{kind}: {name} has no owner; add it to a rule in "
                                   "scripts/capability-inventory.py")
            else:
                self.owner[kind][name] = got


def _set_rules(kind, rules, inv, used):
    seen = {}
    for comp, names in rules.items():
        for n in names:
            if n in seen:
                inv.errors.append(f"{kind}: {n} is claimed by both {seen[n]} and {comp}")
            seen[n] = comp

    def owner_of(name):
        if name in seen:
            used.add(name)
            return seen[name]
        return None
    return seen, owner_of


def _dead(kind, seen, used, inv):
    for n in sorted(set(seen) - used):
        inv.errors.append(f"{kind}: the rule for {n} matches nothing; remove it")


def source_owner(stem):
    for comp, rule in SOURCE_RULES:
        if (rule.match(stem) if hasattr(rule, "match") else stem in rule):
            return comp
    return None


def assign(surfaces):
    """Owners for every surface, and the errors: unowned surfaces, surfaces two
    rules claim, rules that claim nothing."""
    inv = Inventory()

    for kind, rules, key in (("command line", CLI_RULES, None), ("HTTP", HTTP_RULES, None),
                             ("record", RECORD_RULES, "record"),
                             ("TypeScript", TYPESCRIPT_RULES, None)):
        used = set()
        seen, owner_of = _set_rules(kind, rules, inv, used)
        names = surfaces[kind]
        if key == "record":
            inv.claim(kind, names, lambda n: owner_of(n.rsplit(".v", 1)[0]))
        else:
            inv.claim(kind, names, owner_of)
        _dead(kind, seen, used, inv)

    used = set()
    seen, owner_of = _set_rules("Python", PYTHON_RULES, inv, used)
    inv.claim("Python", surfaces["Python"], lambda n: owner_of(surfaces["Python"][n]))
    _dead("Python", seen, used, inv)

    stems = {}
    for path, stem in surfaces["source"].items():
        stems[stem] = source_owner(stem)
    inv.claim("source", surfaces["source"], lambda p: stems[surfaces["source"][p]])
    for comp, rule in SOURCE_RULES:
        hit = [s for s in stems if (rule.match(s) if hasattr(rule, "match") else s in rule)]
        if not hit:
            inv.errors.append(f"source: the {comp} rule {rule} matches no file")
        elif not hasattr(rule, "match"):
            for s in sorted(rule - set(hit)):
                inv.errors.append(f"source: the {comp} rule names {s}, which is not in src/")

    used_name_rules = set()

    def env_owner(name):
        for i, (comp, rule) in enumerate(ENV_NAME_RULES):
            if rule.match(name):
                used_name_rules.add(i)
                return comp
        comps = {source_owner(pathlib.PurePosixPath(f).stem) for f in surfaces["env"][name]}
        if len(comps) == 1 and None not in comps and "shared" not in comps:
            return comps.pop()
        return None
    inv.claim("env", surfaces["env"], env_owner)
    for i, (comp, rule) in enumerate(ENV_NAME_RULES):
        if i not in used_name_rules:
            inv.errors.append(f"env: the {comp} name rule matches no switch; remove it")
    return inv


def collect(root=ROOT):
    return {
        "command line": cli_surfaces(root),
        "HTTP": http_surfaces(root),
        "env": env_surfaces(root),
        "record": record_surfaces(root),
        "Python": python_surfaces(root),
        "TypeScript": typescript_surfaces(root),
        "source": source_surfaces(root),
    }

# ---------------------------------------------------------------- the page

SECTIONS = [
    ("command line", "Command line",
     "Every option `runner --help` declares, the subcommands, and the `runner attest` verbs."),
    ("HTTP", "HTTP routes",
     "Every path `src/server.c` matches. `{id}` stands for a trailing name or id."),
    ("env", "`RUNNER_*` switches",
     "Every switch read through `getenv` or an environment helper, without the `RUNNER_` "
     "prefix. A switch's owner is the component of the file that reads it, unless a name "
     "rule says otherwise. These are outside the compatibility promise "
     "([versioning](versioning.md)); their classes are in the "
     "[deprecation review](deprecation-review-2026-11.md)."),
    ("record", "Record formats", "Every `xyntetik.runner.*` schema identifier, without the prefix."),
    ("Python", "Python client", "The names in each package's `__all__`."),
    ("TypeScript", "TypeScript client", "Every name `clients/typescript/src` exports."),
    ("source", "Source files",
     "Every file under `src/` except the generated kernel headers. `Shared` is the "
     "files that serve more than one component."),
]


def render(surfaces, inv):
    order = COMPONENTS + ["shared"]
    out = [
        "# Capability inventory (R18.2)",
        "",
        "<!-- Generated by scripts/capability-inventory.py --write. Do not edit by hand. -->",
        "",
        "Every public surface of Runner and the one component that owns it. The components are "
        "those of [who owns what](ownership.md). The list is read from the source, and "
        "`scripts/capability-inventory.py --check` fails when a surface has no owner or this "
        "page is out of date.",
        "",
        "| surface | " + " | ".join(TITLES[c] for c in COMPONENTS) + " | total |",
        "|---|" + "---:|" * (len(COMPONENTS) + 1),
    ]
    for kind, title, _ in SECTIONS:
        if kind == "source":
            continue
        owners = inv.owner[kind]
        counts = [sum(1 for v in owners.values() if v == c) for c in COMPONENTS]
        out.append(f"| {title} | " + " | ".join(str(n) if n else "" for n in counts)
                   + f" | {len(owners)} |")
    for kind, title, blurb in SECTIONS:
        owners = inv.owner[kind]
        out += ["", f"## {title} ({len(owners)})", "", blurb, "",
                "| owner | count | surfaces |", "|---|---:|---|"]
        for c in order:
            names = sorted(n for n, v in owners.items() if v == c)
            if not names:
                continue
            out.append(f"| {TITLES[c]} | {len(names)} | "
                       + ", ".join(f"`{n}`" for n in names) + " |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate the page")
    mode.add_argument("--check", action="store_true",
                      help="fail when a surface is unowned or the page is out of date")
    args = ap.parse_args(argv)

    surfaces = collect()
    inv = assign(surfaces)
    if inv.errors:
        print("\n".join(inv.errors), file=sys.stderr)
        return 1
    page = render(surfaces, inv)
    if args.write:
        PAGE.write_text(page, encoding="utf-8", newline="\n")
        print(f"wrote {PAGE.relative_to(ROOT)}")
        return 0
    have = PAGE.read_text(encoding="utf-8").replace("\r\n", "\n") if PAGE.exists() else ""
    if have != page:
        print(f"{PAGE.relative_to(ROOT)} is out of date; run "
              "python3 scripts/capability-inventory.py --write", file=sys.stderr)
        return 1
    total = sum(len(v) for v in inv.owner.values())
    print(f"capability inventory: {total} surfaces, each with one owner")
    return 0


if __name__ == "__main__":
    sys.exit(main())
