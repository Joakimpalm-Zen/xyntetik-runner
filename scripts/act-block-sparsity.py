import numpy as np, sys, collections
path=sys.argv[1]; TARGETS=(0.25,0.40,0.50); B=32
SITES=["attn-in","attn-out","ffn-in","ffn-act"]
def records(path):
    with open(path,"rb") as f:
        while True:
            hdr=np.fromfile(f,dtype=np.int32,count=3)
            if hdr.size<3: return
            site,layer,dim=int(hdr[0]),int(hdr[1]),int(hdr[2])
            v=np.fromfile(f,dtype=np.float32,count=dim)
            if v.size<dim: return
            yield site,layer,v
# pass 1: per (site,layer) magnitude quantiles from a 1/4 subsample of elements
samples=collections.defaultdict(list); n_tok=collections.Counter()
for site,layer,v in records(path):
    m=np.abs(v); samples[(site,layer)].append(m[::4]); n_tok[(site,layer)]+=1
thr={}
for k,lst in samples.items():
    allm=np.concatenate(lst); thr[k]={s: float(np.quantile(allm, s)) for s in TARGETS}
del samples
# pass 2: block-level sparsity at those per-tensor thresholds, plus per-token top-k (dynamic) block sparsity
blk=collections.defaultdict(lambda: {s:[0,0] for s in TARGETS}); dyn=collections.defaultdict(lambda: {s:[0,0] for s in TARGETS}); elem=collections.defaultdict(lambda: {s:[0,0] for s in TARGETS})
for site,layer,v in records(path):
    m=np.abs(v); nb=m.size//B; mb=m[:nb*B].reshape(nb,B).max(axis=1)
    for s in TARGETS:
        t=thr[(site,layer)][s]
        blk[(site,layer)][s][0]+=int((mb<t).sum()); blk[(site,layer)][s][1]+=nb
        elem[(site,layer)][s][0]+=int((m<t).sum()); elem[(site,layer)][s][1]+=m.size
        # dynamic per-token threshold (what RUNNER_ACT_SPARSE applies)
        td=np.partition(m, int(s*m.size))[int(s*m.size)]
        dyn[(site,layer)][s][0]+=int((mb<td).sum()); dyn[(site,layer)][s][1]+=nb
layers=sorted({k[1] for k in blk})
print(f"tokens per tensor: {min(n_tok.values())}..{max(n_tok.values())}; layers {len(layers)}; block {B}")
print(f"{'site':9s} target  elem-sparsity  block-sparsity(static thr)  block-sparsity(per-token thr)   [per-site means over layers; min..max over layers for static]")
for si,name in enumerate(SITES):
    for s in TARGETS:
        es=np.mean([elem[(si,l)][s][0]/elem[(si,l)][s][1] for l in layers])
        bs=[blk[(si,l)][s][0]/max(1,blk[(si,l)][s][1]) for l in layers]
        ds=np.mean([dyn[(si,l)][s][0]/max(1,dyn[(si,l)][s][1]) for l in layers])
        print(f"{name:9s} {s:5.2f}   {es:8.3f}        {np.mean(bs):8.4f}  ({min(bs):.4f}..{max(bs):.4f})        {ds:8.4f}")
