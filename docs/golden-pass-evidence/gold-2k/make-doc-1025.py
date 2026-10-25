"""Write docs/golden-pass-2026-10-25.md from the per-row comparison (compare-1022-1025.json) and the run JSONs. Every table
cell comes from the result files; the prose names only what the tables show. Usage: make-doc-1025.py COMPARE RUN_DIR OUT
[BISECT_SENTENCE]"""
import json
import os
import sys

rows = {r['row']: r for r in json.load(open(sys.argv[1]))}
run, out = sys.argv[2], sys.argv[3]
bisect = sys.argv[4] if len(sys.argv) > 4 else 'These are different build configurations; see the controlled comparison below.'
ORDER = ['smollm2-135m', 'qwen2.5-0.5b', 'qwen3-0.6b', 'qwen3.5-0.8b', 'stablelm-2-1.6b', 'granite-4.0-h-micro', 'granite-4.2-3b',
         'trinity-nano', 'gemma-3-4b-it', 'gemma-4-e2b-it', 'qwen3-4b', 'apertus-8b']
NAME = {'smollm2-135m': 'SmolLM2-135M', 'qwen2.5-0.5b': 'Qwen2.5-0.5B', 'qwen3-0.6b': 'Qwen3-0.6B', 'qwen3.5-0.8b': 'Qwen3.5-0.8B',
        'stablelm-2-1.6b': 'StableLM 2 1.6B', 'granite-4.0-h-micro': 'granite-4.0-h-micro', 'granite-4.2-3b': 'granite-4.2-3B',
        'trinity-nano': 'Trinity Nano', 'gemma-3-4b-it': 'gemma-3-4B', 'gemma-4-e2b-it': 'gemma-4-E2B', 'qwen3-4b': 'Qwen3-4B',
        'apertus-8b': 'Apertus-8B'}
VERD = {'a closer': '1.6.0 closer', 'b closer': None, 'not distinguishable': 'not distinguishable'}


def f(x):
    return ('%.1e' % x).replace('e-0', 'e-') if x < 1e-3 else '%.4f' % x


def p(x):
    return '' if x in (None, -1) else ('%.2g' % x).replace('e-0', 'e-')


def summ(key, which):
    d = json.load(open(os.path.join(run, f'gold1025-{key}-{which}.json')))['summary']
    return d['sides']['a'], d['sides']['b'], d.get('significance', {})


def order():
    for m in ORDER:
        for t in ('bf16', 'served'):
            k = f'{m}-{t}'
            if k in rows:
                yield m, t, k


def row_llama(m, t, k):
    tag = '-forcebos' if m == 'gemma-4-e2b-it' else ''
    a, b, g = summ(f'{m}{tag}-{t}', 'vs-llama')
    v = {'a closer': '1.6.0 closer', 'b closer': 'llama.cpp closer'}.get(g.get('verdict'), g.get('verdict'))
    return (f"| {NAME[m]}, {t} | {f(a['mean_kld_vs_gold'])} | {f(b['mean_kld_vs_gold'])} | {a['margin_qualified_top1_pct']:.2f}% | "
            f"{b['margin_qualified_top1_pct']:.2f}% | {p(g.get('mcnemar_exact_p'))} | {p(g.get('kld_wilcoxon_p'))} | {v} |")


def row_150(m, t, k):
    tag = '-forcebos' if m == 'gemma-4-e2b-it' else ''
    a, b, g = summ(f'{m}{tag}-{t}', 'vs-150')
    v = {'a closer': '1.6.0 closer', 'b closer': '1.5.0 closer'}.get(g.get('verdict'), g.get('verdict'))
    same = a['mean_kld_vs_gold'] == b['mean_kld_vs_gold'] and a['max_kld_vs_gold'] == b['max_kld_vs_gold']
    return (f"| {NAME[m]}, {t} | {f(a['mean_kld_vs_gold'])} | {f(b['mean_kld_vs_gold'])} | {a['margin_qualified_top1_pct']:.2f}% | "
            f"{b['margin_qualified_top1_pct']:.2f}% | {p(g.get('kld_wilcoxon_p'))} | {'identical at every position' if same else v} |")


tr = rows['trinity-nano-served']
closer = sum(1 for r in rows.values() if r['v_new'] == 'a closer')
lvl = sum(1 for r in rows.values() if r['m160'] >= r['mll_new'])
lower = sum(1 for r in rows.values() if r['k160'] < r['kll_new'])
exact = [k for k, r in rows.items() if r['k150_now'] == r['k150_old']]
drift = [k for k, r in rows.items() if r['k150_old'] is not None and r['k150_now'] != r['k150_old']]
served = [r for k, r in rows.items() if k.endswith('-served')]
bf16 = [r for k, r in rows.items() if k.endswith('-bf16')]
cnt = lambda rs, v: sum(1 for r in rs if r['v150'] == v)

L = []
L.append('# Golden Pass: the 1.6.0 source, local Clang build, 2026-10-25\n')
L.append('The README\'s fidelity claim rests on one measurement: each family\'s smallest member scored against the publisher\'s '
         'own implementation in float32, with llama.cpp scored beside Runner on the same file. This page reruns it on 1.6.0. '
         'Same corpus (`tests/fixtures/gold-corpus-2k.txt`), same 2,000 requested positions, same scorer (`scripts/gold-logits.py`, '
         'PR #354), same llama.cpp revision (73a43d1; flash attention off at bf16, on at the served quantization) and the same '
         '23 rows as 2026-10-22, on the CPU. Runner is a clang build of `33c2a08f`, the v1.6.0 merge. Each row is also scored '
         'against the published 1.5.0 asset, the binary the 10-22 page scored, so the combined source/build difference is visible. '
         'Summaries: [`golden-pass-evidence/gold-2k/summaries-2026-10-25.json`](golden-pass-evidence/gold-2k/summaries-2026-10-25.json); '
         'the script that ran: [`gold1025-run.sh`](golden-pass-evidence/gold-2k/gold1025-run.sh).\n')
L.append('## The row that moved against llama.cpp\n')
L.append(f'It comes first, because it changes the README\'s count. On 10-22 Runner read closer to the reference than llama.cpp '
         f'on 20 of 23 rows by the Wilcoxon verdict; on this local Clang build of 1.6.0 it is **{closer} of 23**. The row is **Trinity Nano, served** '
         f'(Q8_0, a mixture-of-experts model): 1.6.0 reads mean KL {f(tr["k160"])} against llama.cpp\'s {f(tr["kll_new"])} '
         f'(10-22: 1.5.0 {f(tr["k150_old"])}, "1.5.0 closer" at p 0.0023), now "not distinguishable" (Wilcoxon p '
         f'{p(tr["w_new"])}). On the chosen token it stays behind llama.cpp, as on 10-22 ({tr["m160"]:.2f}% against '
         f'{tr["mll_new"]:.2f}%). Against 1.5.0 itself the row reads "1.5.0 closer" (p {p(tr["w150"])}), mean KL '
         f'{(tr["k160"] / tr["k150_now"] - 1) * 100:+.1f}%. {bisect}\n')
L.append(f'Over all 23 rows 1.6.0 is closer than llama.cpp on {closer}, level or ahead on the reference\'s chosen token on '
         f'{lvl}, and has the lower mean KL on {lower}.\n')
L.append('## The 1.5.0 asset as an instrument check\n')
L.append(f'The published 1.5.0 asset, scored again beside 1.6.0, reproduces its own 10-22 mean KL to the last digit on all '
         f'{len(exact) + len(drift)} rows. On {len(drift)} of them the raw summaries differ only because they cover different '
         'positions: in the pass against llama.cpp, llama.cpp\'s side drops one position on Qwen2.5-0.5B (position 722, both '
         'tiers) and granite-4.2-3B (813 at bf16, 347 served), on 10-22 as on 1.6.0, so the 10-22 means are over 1,999 or '
         '1,983 positions and the pass against 1.5.0 is over 2,000 or 1,984. Over the common positions the two readings are '
         'equal. The per-position asset and llama.cpp readings match 10-22 exactly; see `golden-pass-evidence/gold-2k/position-audit.json`.\n')
L.append('## The 1.6.0 rows against 1.5.0\n')
L.append(f'On the {len(served)} served rows 1.6.0 and 1.5.0 are not distinguishable on {cnt(served, "not distinguishable")}; '
         f'1.6.0 reads closer on {cnt(served, "a closer")} and 1.5.0 on {cnt(served, "b closer")} (Trinity Nano, above). '
         f'On 10-22 every bf16 row was identical at every position between 1.5.0 and 1.4.2. On 1.6.0 none of the {len(bf16)} '
         f'bf16 rows is: {cnt(bf16, "b closer")} read "1.5.0 closer" by the Wilcoxon test and {cnt(bf16, "not distinguishable")} '
         'are not distinguishable. The four significant bf16 differences are at roughly the 1e-4 scale of mean KL, with the reference\'s chosen token kept at '
         f'every qualified position on those four rows. {bisect}\n')
L.append('## The 1.6.0 rows against llama.cpp\n')
L.append('| row | 1.6.0 mean KL | llama.cpp mean KL | 1.6.0 margin-qualified top-1 | llama.cpp margin-qualified top-1 | McNemar p | Wilcoxon p | verdict |')
L.append('|---|---|---|---|---|---|---|---|')
L += [row_llama(*x) for x in order()]
L.append('\n## The 1.6.0 rows against 1.5.0\n')
L.append('| row | 1.6.0 mean KL | 1.5.0 mean KL | 1.6.0 margin-qualified top-1 | 1.5.0 margin-qualified top-1 | Wilcoxon p | verdict |')
L.append('|---|---|---|---|---|---|---|')
L += [row_150(*x) for x in order()]
L.append('\n## Method notes\n')
L.append('- Host: the lab box\'s CPU route (`--gpu off`), every server at `-t 16 -c 4096`, fresh servers per pass, the GPU '
         'hidden from every server. No CPU pinning (the host\'s cores are shared). Files read from local NVMe copies, each '
         'checked against the sha256 of its read.')
L.append('- gemma-4-E2B is scored on the 10-22 pair itself (the bf16 converted by llama.cpp 73a43d1 and the q4_0 that 1.5.0 '
         'wrote from it), so its rows compare files, not conversions.')
L.append('- The reference is transformers 5.16.1 in float32 on the CPU; for Apertus-8B, bfloat16 storage widened to float32 '
         'at use (`goldref-bf16.py`), as on 10-22.')
L.append('- The published 1.6.0 asset was not scored: it does not start on this host (Ubuntu 24.04, glibc 2.39). It is built '
         'with GCC 15.2 and binds `sqrtf` at GLIBC_2.43; the 1.5.0 asset needs at most GLIBC_2.38 (GCC 13.3). The page scores '
         'a clang build of `33c2a08f` instead.')
L.append('- The pass was paused once at a row boundary for a training run on the same host and resumed; finished rows were '
         'kept, and the remaining row ran on fresh servers as every row does.')
L.append('- Side a (1.6.0) gives the same answer at every position in both passes of every row. On the four rows where '
         'llama.cpp drops a position, the passes\' summaries differ by that one position, not by any answer.')
L.append('\n## Controlled build findings and release status\n')
L.append('A controlled comparison now confirms that **compiler and CPU target each change the score** on Qwen3-0.6B bf16 and Trinity Nano Q8_0. The original Clang native / Conda-sysroot build and published GCC x86-64-v3 build changed multiple factors. With GCC 13.3 and release flags held fixed, v1.5.0 and v1.6.0 give byte-identical complete score JSON on both tested corpus prefixes (1,529 and 1,438 positions). A local v1.5.0 release build is byte-identical to the published asset itself. These probes support build sensitivity and no source regression on those inputs; they do not prove all 23 rows source-equivalent or isolate the full 20-to-19 shift to compiler alone. The earlier bisect sampled eight commits in a 198-commit interval, not all 198; its bad endpoint tested identical, so its printed culprit is invalid. [Build controls, results and limitations](golden-pass-evidence/gold-2k/toolchain/README.md).\n')
L.append('The **published 1.6.0 Linux asset has a confirmed compatibility regression**, requiring GLIBC_2.43 instead of the prior GLIBC_2.38 ceiling. A release-level fix pins Ubuntu 24.04 and checks imported symbol versions before packaging. A local GCC 13.3 / x86-64-v3 rebuild of the v1.6.0 tag passes startup and fixture inference under GLIBC 2.38 and 2.39. This is not a published replacement. The README count remains 20/23 only for the measured published v1.5.0 binary, 19/23 for this measured Clang build; the released v1.6.0 count is unknown. The five-row follow-up requested in the existing review is blocked on a corrected published asset; no v1.6.1 asset was publicly available at this check. No merge or release publication was performed.\n')
L.append('## Source traceability\n')
L.append('[Run provenance](golden-pass-evidence/gold-2k/run-provenance.json) records the exact source, binary/scorer/corpus hashes and reference package versions; [input-files.json](golden-pass-evidence/gold-2k/input-files.json) records read-back hashes of the staged model and HF files. The archived run script fetched main at start; it actually pinned `33c2a08fee8fbd08dd564396466c7c532df12cf2`. Replays must use that commit, not a future main.\n')
L.append('Every table cell is generated from the matching summary in [summaries-2026-10-25.json](golden-pass-evidence/gold-2k/summaries-2026-10-25.json). Gemma E2B uses the explicit `gemma-4-e2b-it-forcebos-{bf16,served}` keys in both passes. [source-files.json](golden-pass-evidence/gold-2k/source-files.json) maps all 46 original result names and SHA256s to archived per-position JSON; only local paths/endpoints were sanitized, and numeric results were preserved. [position-audit.json](golden-pass-evidence/gold-2k/position-audit.json) checks every matched position against the October 22 raw files and verifies side-a repeatability across the two new passes. Scientific notation uses `1.4e-4`. No rows or unfavorable values were removed. [Document generator](golden-pass-evidence/gold-2k/make-doc-1025.py).')
open(out, 'w').write('\n'.join(L) + '\n')
print('written', out)
