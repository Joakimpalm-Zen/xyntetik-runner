"""mlx-lm's decode matvec on Qwen3-4B's shapes: mx.quantized_matmul with
affine 4-bit groups of 64 and bf16 activations (the mlx-community 4-bit
format), one column. Each shape runs 36 independent products per mx.eval (one
per layer, as a decode step has them), so a call's own sync overhead does not
inflate a product's time; median of 20 evals after 5 warm-ups."""
import time, statistics as st, json
import mlx.core as mx
SHAPES = {'q': (2560, 4096), 'k': (2560, 1024), 'v': (2560, 1024), 'o': (4096, 2560),
          'gate': (2560, 9728), 'up': (2560, 9728), 'down': (9728, 2560), 'head': (2560, 151936)}
out = {}
for name, (k, n) in SHAPES.items():
    reps = 1 if name == 'head' else 36
    ws = []
    for _ in range(reps):   # evaluated one at a time: a lazy list would hold every float matrix at once
        q = mx.quantize(mx.random.normal((n, k)).astype(mx.bfloat16), group_size=64, bits=4)
        mx.eval(q); ws.append(q)
    x = mx.random.normal((1, k)).astype(mx.bfloat16)
    mx.eval(ws, x)
    def step():
        ys = [mx.quantized_matmul(x, *ws[i % len(ws)], transpose=True, group_size=64, bits=4) for i in range(reps)]
        mx.eval(ys)
    for _ in range(5): step()
    ts = []
    for _ in range(20):
        t0 = time.perf_counter(); step(); ts.append(time.perf_counter() - t0)
    per = st.median(ts) / reps * 1e3
    bytes_w = n * k / 2 + n * k / 64 * 2 * 2   # 4-bit weights + bf16 scale and bias a group
    del ws
    out[name] = {'ms': round(per, 4), 'GB_s': round(bytes_w / (per / 1e3) / 1e9, 1)}
print(json.dumps(out), flush=True)
per_step = sum(out[s]['ms'] * 36 for s in SHAPES if s != 'head') + out['head']['ms']
print(f'matvecs a decode step (36 layers x 7 + head): {per_step:.2f} ms')
