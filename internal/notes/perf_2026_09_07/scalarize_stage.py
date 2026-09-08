import re, ast, sys, math
whole = open(sys.argv[1]).read()
funcs = {}
for m in re.finditer(r'^static (?:__attribute__\(\(noinline\)\) |inline )void (\w+)\(([^)]*)\) \{\n(.*?)^\}\n', whole, flags=re.S|re.M):
    funcs[m.group(1)] = (m.group(2), m.group(3).splitlines())
target = sys.argv[4] if len(sys.argv) > 4 else list(funcs)[-1]
sig, body = funcs[target]
src = [f'void {target}({sig}) {{'] + body + ['}']
outname = sys.argv[2] if len(sys.argv) > 2 else 'folded'
seed_arg = sys.argv[3] if len(sys.argv) > 3 else 'fwd_z'
seed_vals = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

class Sym:
    __slots__ = ('op','args','val','id')
    def __init__(s, op, args=(), val=None):
        s.op=op; s.args=args; s.val=val
    def isc(s): return s.op=='const'
table = {}
counter=[0]
def mk(op,*args,val=None):
    key=(op,)+tuple(id(a) for a in args)+((val,) if op in('const','sym') else ())
    n=table.get(key)
    if n is None:
        n=Sym(op,args,val); n.id=counter[0]; counter[0]+=1; table[key]=n
    return n
def C(v): return mk('const',val=float(v))
def wrap(x): return x if isinstance(x,Sym) else C(x)
def add(a,b):
    a,b=wrap(a),wrap(b)
    if a.isc() and b.isc(): return C(a.val+b.val)
    if a.isc() and a.val==0: return b
    if b.isc() and b.val==0: return a
    if b.op=='neg': return sub(a,b.args[0])
    if a.op=='neg': return sub(b,a.args[0])
    return mk('add',a,b)
def sub(a,b):
    a,b=wrap(a),wrap(b)
    if a.isc() and b.isc(): return C(a.val-b.val)
    if b.isc() and b.val==0: return a
    if a.isc() and a.val==0: return neg(b)
    if b.op=='neg': return add(a,b.args[0])
    return mk('sub',a,b)
def mul(a,b):
    a,b=wrap(a),wrap(b)
    if a.isc() and b.isc(): return C(a.val*b.val)
    if a.isc() and a.val==0: return C(0.0)
    if b.isc() and b.val==0: return C(0.0)
    if a.isc() and a.val==1: return b
    if b.isc() and b.val==1: return a
    if a.isc() and a.val==-1: return neg(b)
    if b.isc() and b.val==-1: return neg(a)
    if a.op=='neg' and b.op=='neg': return mul(a.args[0],b.args[0])
    return mk('mul',a,b)
def div(a,b):
    a,b=wrap(a),wrap(b)
    if a.isc() and b.isc(): return C(a.val/b.val)
    if a.isc() and a.val==0: return C(0.0)
    if b.isc() and b.val==1: return a
    return mk('div',a,b)
def neg(a):
    a=wrap(a)
    if a.isc(): return C(-a.val)
    if a.op=='neg': return a.args[0]
    return mk('neg',a)
def fn(name):
    def f(a):
        a=wrap(a)
        if a.isc(): return C(getattr(math,name)(a.val))
        return mk(name,a)
    return f
def pw(a,b):
    a,b=wrap(a),wrap(b)
    if a.isc() and b.isc(): return C(a.val**b.val)
    if b.isc() and b.val==2: return mul(a,a)
    return mk('pow',a,b)
Sym.__add__=add; Sym.__radd__=lambda s,o: add(o,s); Sym.__sub__=sub; Sym.__rsub__=lambda s,o: sub(o,s)
Sym.__mul__=mul; Sym.__rmul__=lambda s,o: mul(o,s); Sym.__truediv__=div; Sym.__rtruediv__=lambda s,o: div(o,s); Sym.__neg__=neg

class Alias:  # pointer alias into a list with offset
    def __init__(s,base,off): s.base=base; s.off=off
    def __add__(s,n): return Alias(s.base,s.off+int(n))
    __radd__=__add__
    def __getitem__(s,i): return s.base[s.off+i]
    def __setitem__(s,i,v): s.base[s.off+i]=v

env={'w':[None]*100000,'cos':fn('cos'),'sin':fn('sin'),'tanh':fn('tanh'),'exp':fn('exp'),'sqrt':fn('sqrt'),'log':fn('log'),'pow':pw}
inputs={}
def declare_input(name,n):
    inputs[name]=n
    env[name]=[mk('sym',val=f'{name}[{i}]') for i in range(n)]
# signature
sig=src[0]
params=re.findall(r'(const )?double\* (\w+)',sig)
insizes={}
outputs=[]
for const,name in params:
    if name=='w': continue
    if name==seed_arg: env[name]=[C(v) for v in seed_vals]
    elif const: declare_input(name,insizes.get(name,4096))
    else: env[name]=[None]*4096; outputs.append(name)

def cexpr(e):
    return e.replace('&&','and')
def run(lines):
    i=0
    while i<len(lines):
        l=lines[i].strip(); i+=1
        if not l or l.startswith('(void)') or l=='}': continue
        m=re.match(r'const double\* (\w+) = (\w+)(?: \+ (\d+))?;',l)
        if m:
            env[m.group(1)]=Alias(env[m.group(2)] if isinstance(env[m.group(2)],list) else env[m.group(2)].base, int(m.group(3) or 0)+(0 if isinstance(env[m.group(2)],list) else env[m.group(2)].off)); continue
        m=re.match(r'static const (double|int64_t) (\w+)\[\d+\] = \{([^}]*)\};',l)
        if m:
            vals=[float(x) if m.group(1)=='double' else int(x) for x in m.group(3).split(',')]
            env[m.group(2)]=[C(v) for v in vals] if m.group(1)=='double' else vals; continue
        m=re.match(r'double (\w+)\[(\d+)\];',l)
        if m: env[m.group(1)]=[None]*int(m.group(2)); continue
        m=re.match(r'double\* (\w+) = w \+ (\d+);',l)
        if m: env[m.group(1)]=Alias(env['w'],int(m.group(2))); continue
        m=re.match(r'(\w+)\((.*)\);$',l)
        if m and m.group(1) in funcs:
            csig,cbody=funcs[m.group(1)]
            depth=0; args=[]; cur=''
            for ch in m.group(2):
                if ch=='(' : depth+=1
                if ch==')' : depth-=1
                if ch==',' and depth==0: args.append(cur.strip()); cur=''
                else: cur+=ch
            args.append(cur.strip())
            penv={k:(Alias(v,0) if isinstance(v,list) else v) for k,v in env.items() if isinstance(v,(list,Alias,int))}
            args=[None if a=='NULL' else eval(a,{},penv) for a in args]
            saved=dict(env)
            for (pconst,pname),a in zip(re.findall(r'(const )?double\* (\w+)',csig),args):
                env[pname]=Alias(env['w'],0) if a is None else a
            run(cbody)
            for k in list(env):
                if k not in saved: del env[k]
            for k in saved: env[k]=saved[k]
            continue
        m=re.match(r'for \(long long (\w+) = 0; \1 < (\d+); \+\+\1\) \{',l)
        if m:
            body=[]; depth=1
            while depth:
                b=lines[i]; i+=1
                if b.strip().endswith('{'): depth+=1
                if b.strip()=='}': depth-=1
                if depth: body.append(b)
            for v in range(int(m.group(2))):
                env[m.group(1)]=v; run(body)
            continue
        m=re.match(r'(\w+)\[(.+?)\] = (.*);$',l)
        if m:
            arr,idx,rhs=m.groups()
            # find matching bracket for idx: re-split properly
            lhs,rhs=l[:-1].split(' = ',1)
            arr=lhs[:lhs.index('[')]; idx=lhs[lhs.index('[')+1:-1]
            ii=eval(idx,env); val=eval(cexpr(rhs),env)
            env[arr][ii]=wrap(val); continue
        raise SystemExit('unhandled: '+l)
run(src[1:])
# emit
order=[]; seen=set()
def visit(n):
    if n.id in seen: return
    seen.add(n.id)
    for a in n.args: visit(a)
    order.append(n)
outs={}
for o in outputs:
    for j,v in enumerate(env[o]):
        if v is not None: outs[(o,j)]=v; visit(v)
names={}
lines=[]
cnt={}
for n in order:
    if n.op=='sym': names[n.id]=n.val; continue
    if n.op=='const': names[n.id]=repr(n.val); continue
    cnt[n.op]=cnt.get(n.op,0)+1
    a=[names[x.id] for x in n.args]
    e={'add':lambda:f'{a[0]} + {a[1]}','sub':lambda:f'{a[0]} - {a[1]}','mul':lambda:f'{a[0]} * {a[1]}','div':lambda:f'{a[0]} / {a[1]}','neg':lambda:f'-{a[0]}','pow':lambda:f'pow({a[0]}, {a[1]})'}.get(n.op,lambda:f'{n.op}({a[0]})')()
    names[n.id]=f'a{n.id}'
    lines.append(f'  double a{n.id} = {e};')
argl=', '.join(f'const double* {k}' for k in inputs)+', '+', '.join(f'double* {o}' for o in outputs)
body='\n'.join(lines)+'\n'+'\n'.join(f'  {o}[{j}] = {names[v.id]};' for (o,j),v in outs.items())
open(outname+'.c','w').write(f'#include <math.h>\nstatic inline void {outname}({argl}) {{\n{body}\n}}\n')
print('ops:',cnt, 'total', sum(cnt.values()), 'outputs', len(outs), file=sys.stderr)
