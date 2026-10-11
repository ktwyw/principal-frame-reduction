"""Digitise marker positions from the 600-dpi bitmap figures of Daivis, Matin & Todd, JNNFM 111, 1 (2003):
Fig. 1 (eta), Fig. 2 (Psi1), Fig. 4 (eta1).  Strain rates are snapped to the simulation grid II = {1,2,5} x 10^n
(II = 2 gdot^2 = 8 eps^2), which removes abscissa error; ordinates come from the axis frame."""
import json, numpy as np
from PIL import Image
from scipy import ndimage as ndi
def frame(img):
    rows = img.sum(1); cols = img.sum(0); ry = np.where(rows > 0.6*rows.max())[0]; cx = np.where(cols > 0.6*cols.max())[0]
    return cx.min(), cx.max(), ry.min(), ry.max()
def blobs(img, x0, x1, y0, y1, legend, k):
    inner = np.zeros_like(img); inner[y0+6:y1-5, x0+6:x1-5] = img[y0+6:y1-5, x0+6:x1-5]; lx0, ly1 = legend; inner[:ly1, lx0:] = 0
    filled = ndi.binary_fill_holes(inner); core = ndi.binary_opening(filled, structure=np.ones((k, k))) if k > 1 else filled
    lab, n = ndi.label(core); out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        m = lab[sl] == i; h, w = m.shape
        if not (16 <= w <= 45 and 16 <= h <= 70): continue
        cy, cx = ndi.center_of_mass(m); rw = m.sum(1)/m.sum(1).max(); q = max(2, h//5)
        out.append(dict(px=float(sl[1].start + cx), py=float(sl[0].start + cy), top_py=float(sl[0].start), bot_py=float(sl[0].stop), w=int(w), h=int(h), ink=float(inner[sl][m].mean()),
                        top=float(rw[:q].mean()), bot=float(rw[-q:].mean()), merged=bool(h > 1.35*w)))
    return out
def shape(b):
    if b['merged']: return 'merged'
    t, bo = b['top'], b['bot']
    if t > 0.85 and bo > 0.85: return 'square'
    if bo - t > 0.3: return 'triangle'
    if t < 0.5 and bo < 0.5: return 'diamond'
    return 'circle'
grid = np.array(sorted({np.sqrt(m*10.0**n/2) for n in range(-9, 2) for m in (1, 2, 5)}))      # shear-rate grid from II = 2 gdot^2
CFG = {'fig1_eta.png': dict(yr=(0, 2), legend=(0.69, 0.66), k=1, anchors=(2.236e-4, 2.236), div=1.0),
       'fig2_psi1.png': dict(yr=(-1, 5), legend=(0.74, 0.62), k=5, anchors=(2.236e-4, 2.236), div=1.0),
       'fig4_eta1.png': dict(yr=(0, 3), legend=(1.1, 0.0), k=5, anchors=(1.118e-4, 1.118), div=2.0)}
res = {}
for f, c in CFG.items():
    img = np.array(Image.open(f).convert('L')) > 127; x0, x1, y0, y1 = frame(img); lg = (int(x0 + c['legend'][0]*(x1-x0)), int(y0 + c['legend'][1]*(y1-y0)))
    bl = [b for b in blobs(img, x0, x1, y0, y1, lg, c['k']) if b['py'] > y0 + 0.06*(y1 - y0) or b['px'] < x0 + 0.8*(x1 - x0)]   # drop artefacts in the top-right corner
    pxs = np.array([b['px'] for b in bl]); pa, pb = pxs.min(), pxs.max()       # leftmost / rightmost markers are the anchors
    for b in bl:
        lx = np.log10(c['anchors'][0]) + (b['px'] - pa)/(pb - pa)*np.log10(c['anchors'][1]/c['anchors'][0]); g = grid/c['div']; j = np.argmin(abs(np.log10(g) - lx))
        b['x'] = float(g[j]); b['xerr'] = float(10**lx/g[j] - 1); yy = lambda p: float(10**(c['yr'][1] - (p - y0)/(y1 - y0)*(c['yr'][1] - c['yr'][0])))
        b['y'] = yy(b['py']); b['ytop'] = yy(b['top_py'] + b['w']/2); b['ybot'] = yy(b['bot_py'] - b['w']/2); b['shape'] = shape(b); b['open'] = b['ink'] < 0.8
    res[f] = bl; print(f'--- {f}: {len(bl)} blobs; max |x snap error| = {max(abs(b["xerr"]) for b in bl)*100:.1f}%')
    for x in sorted({b['x'] for b in bl}):
        print(f'  x={x:.3e}: ' + ' | '.join(f"{b['shape'][:4]}{'o' if b['open'] else 's'} {b['y']:.4g}" + (f" [{b['ybot']:.4g}..{b['ytop']:.4g}]" if b['merged'] else '') for b in sorted([b for b in bl if b['x'] == x], key=lambda b: -b['y'])))
json.dump(res, open('digitized_raw.json', 'w'))
