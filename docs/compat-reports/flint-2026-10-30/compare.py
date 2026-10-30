import json, math, sys
r=json.load(open(sys.argv[1])); s=json.load(open(sys.argv[2]))
assert s['tokens'][:len(r['ids'])]==r['ids']
klds=[]; maxd=0; agree=0; n=len(s['top'])
for p in range(n):
    a={i:v for i,v in r['top'][p]}; b={int(i):float(v) for i,v in s['top'][p]}
    shared=set(a)&set(b); maxd=max(maxd, max(abs(a[k]-b[k]) for k in shared))
    keys=set(a)|set(b); fa=min(a.values())-1; fb=min(b.values())-1
    pa={k:math.exp(a.get(k,fa)) for k in keys}; pb={k:math.exp(b.get(k,fb)) for k in keys}
    za=sum(pa.values()); zb=sum(pb.values())
    klds.append(sum(pa[k]/za*math.log((pa[k]/za)/(pb[k]/zb)) for k in keys))
    agree+= max(a,key=a.get)==max(b,key=b.get)
print('positions %d  mean KLD %.3g  max KLD %.3g  max |dlogp| %.3g  top-1 %.1f%%' % (n, sum(klds)/n, max(klds), maxd, 100*agree/n))
