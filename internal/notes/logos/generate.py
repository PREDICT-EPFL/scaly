import bisect, math, sys

# The casing is the DejaVu Sans 'S', the same face as the letters, placed at its own
# advance width in front of them. Scales are laid along a hand-placed centreline and the
# glyph outline clips them, so the silhouette is exactly the font's.
K = 0.1000977                     # font units -> drawing units, as used for the letters
SX, BASE = 90.0, 210.0            # S origin; 'cali' starts one advance (1300 units) later
import json
import os
FONT = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "glyphs.json")))       # DejaVu Sans Bold: S, c, a, l, i
GLYPH = FONT["S"]["d"]
_rx = lambda x: 147 + (x - 135) * (1325 - 147) / (1186 - 135)   # regular-S spine -> bold-S bbox
SPINE = [(60,200),(175,170),(380,90),(614,53),(850,110),(1010,250),(1080,405),(1040,560),(880,680),
         (660,770),(450,870),(300,980),(236,1102),(300,1290),(470,1400),(659,1438),
         (860,1420),(1000,1385),(1060,1345),(1170,1290)]
P = [(SX + _rx(x)*K, BASE - y*K) for x, y in SPINE]

def catmull(pts, n=16):
    res=[]; ext=[pts[0]]+pts+[pts[-1]]
    for i in range(1,len(ext)-2):
        p0,p1,p2,p3=ext[i-1],ext[i],ext[i+1],ext[i+2]
        for k in range(n):
            t=k/n; t2=t*t; t3=t2*t
            res.append(tuple(0.5*(2*p1[j]+(-p0[j]+p2[j])*t+(2*p0[j]-5*p1[j]+4*p2[j]-p3[j])*t2+(-p0[j]+3*p1[j]-3*p2[j]+p3[j])*t3) for j in (0,1)))
    res.append(pts[-1]); return res

pts=catmull(P)
S=[0.0]
for a,b in zip(pts,pts[1:]): S.append(S[-1]+math.hypot(b[0]-a[0],b[1]-a[1]))
L=S[-1]

def frame(s):
    s=max(0.0,min(L,s)); i=max(1,min(bisect.bisect(S,s),len(pts)-1))
    a,b=pts[i-1],pts[i]; seg=S[i]-S[i-1]; u=(s-S[i-1])/seg if seg else 0.0
    dx,dy=b[0]-a[0],b[1]-a[1]; n=math.hypot(dx,dy)
    return a[0]+dx*u, a[1]+dy*u, dx/n, dy/n, -dy/n, dx/n

def pos(s,t):
    x,y,_,_,nx,ny=frame(s); return (x+nx*t,y+ny*t)

def curvature(s,h=2.0):
    _,_,x0,y0,_,_=frame(s-h); _,_,x1,y1,_,_=frame(s+h); return (x0*y1-y0*x1)/(2*h)

W, GAP = 48.0, 3.2                # band a little wider than the glyph stroke, so the clip always cuts
COLORS=["#111b20","#182226","#1e282b"]
def lozenge(s,t,a,b,k=0.64):
    c=lambda ds,dt: pos(s+ds,t+dt)
    P0,P1,P2,P3=c(-a,0),c(0,-b),c(a,0),c(0,b)
    C=[c(-a*k,-b*k),c(a*k,-b*k),c(a*k,b*k),c(-a*k,b*k)]
    d=f"M{P0[0]:.1f} {P0[1]:.1f}"
    for ctl,e in zip(C,(P1,P2,P3,P0)): d+=f"Q{ctl[0]:.1f} {ctl[1]:.1f} {e[0]:.1f} {e[1]:.1f}"
    return d+"Z"

p=W/2; b=p/2-GAP/2; a=1.35*b
cells=[]; s=-a; row=0; ci=0
while s<L+a:
    kap=curvature(s)
    for t in ([-p,0,p] if row%2==0 else [-0.5*p,0.5*p]):
        if 1-kap*t>0.2:
            cells.append(f'<path d="{lozenge(s,t,a,b)}" fill="{COLORS[ci%3]}"/>'); ci+=1
    s+=a+GAP; row+=1

_x = SX + FONT["S"]["adv"] * K
_parts = []
for ch in "cali":
    _parts.append(f'<path d="{FONT[ch]["d"]}" transform="translate({_x:.3f} {BASE}) scale({K} -{K})"/>')
    _x += FONT[ch]["adv"] * K
letters = '<g fill="#151e22">\n' + "\n".join(_parts) + '\n</g>'
RIGHT = _x

HERE = os.path.dirname(os.path.abspath(__file__))
def write(name, body, x0, y0, x1, y1, title):
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.0f} {y0:.0f} {x1-x0:.0f} {y1-y0:.0f}" role="img" aria-labelledby="title">
<title id="title">{title}</title>
<defs><clipPath id="glyph"><path d="{GLYPH}" transform="translate({SX} {BASE}) scale({K} -{K})"/></clipPath></defs>
<g clip-path="url(#glyph)">
{chr(10).join(cells)}
</g>
{body}
</svg>
'''
    open(os.path.join(HERE, name), "w").write(svg)

top, bottom = BASE - 1556*K - 16, BASE + 29*K + 16
write("scali-wordmark.svg", letters, SX + 147*K - 16, top, RIGHT - 172*K + 16, bottom,
      "Scali: the S filled with charcoal snake scales")
write("scali-mark.svg", "", SX + 147*K - 16, top, SX + 1325*K + 16, bottom,
      "Scali mark: an S filled with charcoal snake scales")
print("cells", len(cells))
