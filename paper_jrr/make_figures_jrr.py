"""Figures 1-3 of the JRR manuscript, drawn from released result files and results_jrr/derived/ (no hand-placed data).
Usage: python paper_jrr/make_figures_jrr.py .   -> paper_jrr/figures/fig{1,2,3}.pdf (+png)"""
import json, os, sys, csv, collections
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'
D = os.path.join(ROOT, 'results_jrr', 'derived'); OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.dpi': 300, 'axes.linewidth': 0.7})
COL = {'noise': '#6b6b6b', 'bias': '#0072B2', 'gain': '#E69F00', 'drift': '#D55E00'}
BB = ['LSTM', 'Transformer']
def letter(ax, s): ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')

# ------------------------------------------------------------------ Figure 1: evaluation-to-decision flow
fig, ax = plt.subplots(figsize=(7.2, 3.7)); ax.set_xlim(-3, 103); ax.set_ylim(-3, 51); ax.axis('off')
def box(x, y, w, h, txt, fc='#f2f2f2', ec='#444444', fs=6.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=1.0', fc=fc, ec=ec, lw=0.8))
    ax.text(x + w/2, y + h/2, txt, ha='center', va='center', fontsize=fs, linespacing=1.3)
def arrow(x0, y0, x1, y1, ls='-'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, lw=0.9, color='#333333', linestyle=ls, shrinkA=0, shrinkB=0))
ax.text(0, 48.2, 'Training time', fontsize=7.5, fontweight='bold', va='center'); ax.text(17, 48.2, 'clean data; engine-level split of the official training file', fontsize=6.5, va='center', color='#444444')
box(0, 36, 22, 9, 'Fitting engines (60%)\nweights and\nmin–max normaliser', fc='#e8f0f8')
box(25, 36, 22, 9, 'Validation engines (20%)\ncheckpoint selection\nonly', fc='#e8f0f8')
box(50, 36, 26, 9, 'Calibration engines (20%)\nresidual scale $\\sigma_{res}$ and\nconformal quantile', fc='#e8f0f8')
ax.text(79, 40.5, 'Official test file used\nonly after weights and\nquantiles are frozen', fontsize=6.3, va='center', color='#444444', style='italic')
ax.text(0, 29.5, 'Evaluation time', fontsize=7.5, fontweight='bold', va='center'); ax.text(19.5, 29.5, 'one decision per engine', fontsize=6.5, va='center', color='#444444')
box(0, 16.5, 15, 9, 'Test engine:\nlast window,\n14 or 15 sensor\nchannels', fc='#f2f2f2')
box(18.5, 16.5, 19, 9, 'Sensor degradation\n(sensor channels only)\nbias, drift, gain, noise', fc='#fbe5d6', ec='#D55E00')
box(41, 16.5, 21, 9, 'LSTM or Transformer;\nfive interval mechanisms:\n$\\hat\\mu$, $\\hat\\sigma$ or conformal $q$', fc='#e8f0f8')
box(65.5, 16.5, 14, 9, 'One-sided 95%\nlower bound $b$', fc='#f2f2f2')
box(83, 16.5, 17, 9, 'Trigger if $b\\leq L$,\n$L$ = 10, 20, 30\ncycles', fc='#f2f2f2')
for x0, x1 in [(15.4, 18.1), (37.9, 40.6), (62.4, 65.1), (79.9, 82.6)]: arrow(x0, 21, x1, 21)
arrow(36, 35.6, 49, 26.1, ls='--'); arrow(63, 35.6, 55, 26.1, ls='--')
ax.plot([91.5, 91.5], [16.1, 12.5], color='#333333', lw=0.9); ax.plot([15, 91.5], [12.5, 12.5], color='#333333', lw=0.9)
box(0, 0, 30, 8.2, 'Unrecognised need $r_{late}$:\ntrue RUL $\\leq L$, not triggered', fc='#fde9e7', ec='#b03a2e')
box(34, 0, 30, 8.2, 'Premature trigger $r_{early}$:\ntrue RUL $> L+20$, triggered', fc='#fde9e7', ec='#b03a2e')
box(68, 0, 32, 8.2, 'Cost = $c_{late}\\,r_{late} + r_{early}$ with\n$c_{late}$ = 5, 20, 100; other states\ncarry no cost', fc='#f2f2f2')
for xc in (15, 49, 84): arrow(xc, 12.5, xc, 8.6)
fig.savefig(os.path.join(OUT, 'fig1.pdf'), bbox_inches='tight', pad_inches=0.15); fig.savefig(os.path.join(OUT, 'fig1.png'), bbox_inches='tight', pad_inches=0.15, dpi=200); plt.close(fig)

# ------------------------------------------------------------------ Figure 2: coverage versus f_oob
rows = list(csv.DictReader(open(os.path.join(D, 'foob_coverage_points.csv'))))
def series(bb, deg, ds):
    r = sorted([(float(x['level_pct_range']), float(x['f_oob']), float(x['picp'])) for x in rows if x['backbone'] == bb and x['degradation'] == deg and x['dataset'] == ds and x['mechanism'] == 'NLL'])
    return np.array(r)
FLOOR = 3e-6
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), sharey=True)
panels = [(0, 0, 'LSTM', ['FD002', 'FD004']), (0, 1, 'Transformer', ['FD002', 'FD004']), (1, 0, 'LSTM', ['FD001', 'FD003']), (1, 1, 'Transformer', ['FD001', 'FD003'])]
MK = ['o', 's']
for (i, j, bb, dss) in panels:
    ax = axes[i, j]
    for deg in ['noise', 'bias', 'gain', 'drift']:
        for k, ds in enumerate(dss):
            s = series(bb, deg, ds); x = np.maximum(s[:, 1], FLOOR)
            ax.plot(x, s[:, 2], '-', color=COL[deg], lw=0.8, alpha=0.8)
            ax.plot(x, s[:, 0] * 0 + s[:, 2], MK[k], color=COL[deg], ms=4.2, mfc=COL[deg] if k == 0 else 'white', mew=0.9, zorder=3)
    ax.axhline(0.9, color='#888888', lw=0.7, ls=(0, (4, 3)))
    ax.set_xscale('log'); ax.set_ylim(-0.02, 1.0)
    if i == 0:
        ax.set_xlim(3e-3, 0.6); ax.set_xticks([1e-2, 1e-1]); ax.set_xticklabels(['$10^{-2}$', '$10^{-1}$']); ax.minorticks_off()
    else:
        ax.set_xlim(1.5e-6, 1.3); ax.set_xticks([FLOOR, 1e-4, 1e-3, 1e-2, 1e-1, 1]); ax.set_xticklabels(['0', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '1'])
    ax.set_title(f"{bb}, {'six-condition' if i == 0 else 'single-condition'} sub-datasets", loc='left', fontsize=8)
    ax.set_xlabel('Fraction of normalised inputs outside [−1, 1]  ($f_{oob}$)')
    if j == 0: ax.set_ylabel('Coverage of the 90% interval (PICP)')
    ax.margins(0.02)
    if i == 0:
        ax.axvline(0.10, color='#333333', lw=0.6, ls=':')
        ax.text(0.105, 0.97, 'matched $f_{oob}$ = 10%', fontsize=6.5, va='top', ha='left', color='#333333')
    else:
        if j == 0:
            ax.text(2.5e-5, 0.50, 'drift: $f_{oob}$ < $10^{-5}$,\ncoverage 0.1–0.6', fontsize=6.5, color=COL['drift'], va='center')
            ax.text(2.5e-3, 0.52, 'gain: $f_{oob}$ ≥ 14% at the\nlowest level, coverage < 0.45', fontsize=6.5, color='#8a5a00', va='center')
axes[0, 0].text(3.3e-3, 0.915, 'nominal 0.90', fontsize=6.5, color='#666666', va='bottom')
h = [plt.Line2D([], [], color=COL[d], lw=1.4, label=d.capitalize() if d != 'noise' else 'Additive noise') for d in ['noise', 'bias', 'gain', 'drift']]
h += [plt.Line2D([], [], color='k', marker='o', ls='', ms=4.2, label='FD002 / FD001'), plt.Line2D([], [], color='k', marker='s', mfc='white', ls='', ms=4.2, label='FD004 / FD003')]
fig.legend(handles=h, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.03), handlelength=1.4, columnspacing=1.4)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(os.path.join(OUT, 'fig2.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(OUT, 'fig2.png'), bbox_inches='tight', dpi=200); plt.close(fig)

# ------------------------------------------------------------------ Figure 3: trigger errors and cost sensitivity
boot = json.load(open(os.path.join(D, 'decision_engine_bootstrap.json'))); rk = json.load(open(os.path.join(D, 'rank_stability.json')))
DS = ['FD001', 'FD002', 'FD003', 'FD004']; CONDS = [('clean', 'Clean', '#9a9a9a'), ('gaussian1pct', 'Noise 1%', '#6b6b6b'), ('bias5pct', 'Bias 5%', COL['bias']), ('drift5pct', 'Drift 5%', COL['drift'])]
def b(bb, ds, c, key, L=20, m='NLL'): return boot[f'{bb}|{ds}|{c}|{m}|L{L}'][key]
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6))
for ai, (key, ttl) in enumerate([('prem', 'Premature triggers  $r_{early}$  (L = 20)'), ('unrec', 'Unrecognised maintenance needs  $r_{late}$  (L = 20)')]):
    ax = axs[0, ai]; w = 0.19
    for ci, (c, cn, col) in enumerate(CONDS):
        for bi, bb in enumerate(BB):
            for di, ds in enumerate(DS):
                v = b(bb, ds, c, key); x = di + (ci - 1.5) * w * 1.15 + (bi - 0.5) * 0.085
                ax.errorbar(x, 100 * v[0], yerr=[[100 * (v[0] - v[1])], [100 * (v[2] - v[0])]], fmt='o' if bi == 0 else 's', ms=3.2, color=col, mfc=col if bi == 0 else 'white', mew=0.8, elinewidth=0.7, capsize=0, label=cn if (di == 0 and bi == 0) else None)
    ax.set_xticks(range(4)); ax.set_xticklabels(DS); ax.set_ylabel('% of evaluated engines'); ax.set_title(ttl, loc='left', fontsize=8); ax.margins(x=0.04)
    letter(ax, 'ab'[ai])
    if ai == 0: ax.legend(frameon=False, loc='upper left', fontsize=6.5, handlelength=1.0, borderpad=0.2, labelspacing=0.25)
axs[0, 1].plot([], [], 'ko', ms=3.2, label='LSTM'); axs[0, 1].plot([], [], 'ks', ms=3.2, mfc='white', label='Transformer'); axs[0, 1].legend(frameon=False, loc='upper left', fontsize=6.5, handlelength=1.0, borderpad=0.2, labelspacing=0.25)
# (c) lead-time sensitivity, six-condition
ax = axs[1, 0]; SIX = [(bb, ds) for bb in BB for ds in ['FD002', 'FD004']]; Ls = [10, 20, 30]
for c, cn, col, key, ls in [('drift5pct', 'Drift: premature', COL['drift'], 'prem', '-'), ('bias5pct', 'Bias: premature', COL['bias'], 'prem', '-'), ('bias5pct', 'Bias: unrecognised', COL['bias'], 'unrec', '--')]:
    arr = np.array([[100 * b(bb, ds, c, key, L)[0] for L in Ls] for bb, ds in SIX])
    for r_ in arr: ax.plot(Ls, r_, ls, color=col, lw=0.7, alpha=0.45)
    ax.plot(Ls, arr.mean(0), ls, color=col, lw=1.8, marker='o', ms=3.5)
    ax.text(30.6, arr.mean(0)[-1], cn, fontsize=6.5, color=col, va='center')
ax.set_xlim(8.5, 41); ax.set_xticks(Ls); ax.set_xlabel('Lead time $L$ (cycles)'); ax.set_ylabel('% of evaluated engines'); ax.set_title('Lead-time dependence, six-condition sub-datasets', loc='left', fontsize=8); letter(ax, 'c')
# (d) rank reversals by condition
ax = axs[1, 1]; cnt = collections.defaultdict(lambda: [0, 0, 0])
for k, v in rk.items():
    c = k.split('|')[2]; cnt[c][0] += v['reversal']; cnt[c][1] += v['top_change']; cnt[c][2] += 1
xs = np.arange(4); wd = 0.36
for i, (c, cn, col) in enumerate(CONDS):
    ax.bar(i - wd/2, cnt[c][0], wd, color=col, edgecolor=col); ax.bar(i + wd/2, cnt[c][1], wd, color='white', edgecolor=col, hatch='////', lw=0.8)
    ax.text(i - wd/2, cnt[c][0] + 0.5, str(cnt[c][0]), ha='center', fontsize=6.5); ax.text(i + wd/2, cnt[c][1] + 0.5, str(cnt[c][1]), ha='center', fontsize=6.5)
ax.set_xticks(xs); ax.set_xticklabels([c[1] for c in CONDS], fontsize=6.5); ax.set_ylim(0, 27); ax.set_ylabel('Cases out of 24')
ax.set_title('Mechanism ranking changes across cost ratios 5/20/100', loc='left', fontsize=8); letter(ax, 'd')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc='#777777', ec='#777777', label='full ranking differs'), Patch(fc='white', ec='#777777', hatch='////', label='top-ranked mechanism differs')], frameon=False, loc='upper right', fontsize=6.5, handlelength=1.4)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig3.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(OUT, 'fig3.png'), bbox_inches='tight', dpi=200); plt.close(fig)
print('figures written to', OUT)
