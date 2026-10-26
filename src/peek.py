import json,sys,collections
for f in sys.argv[1:]:
    rows=[json.loads(l) for l in open(f)]
    c=collections.defaultdict(lambda:[0,0,0,0])
    for r in rows:
        if r['condition'].startswith('monitor'): continue
        t=r['ids'][r['target']]; x=c[r['condition']]
        x[0]+=1; x[1]+= r['guess']==t; x[2]+= r['decisions'][r['target']]=='approve'; x[3]+= 'missing' in r['decisions']
    print(f, {k:f"n{v[0]} rec{v[1]} appT{v[2]} miss{v[3]}" for k,v in c.items()})
