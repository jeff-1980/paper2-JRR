"""JRR revision figures. Run: python rev_figs.py <res_sd dir> <perchannel_sd.json> <outdir>"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RES, PC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.6,
                     'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.labelcolor': '#0b0b0b', 'axes.spines.top': False, 'axes.spines.right': False})
BLUE, ORANGE, AQUA, VIOLET, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#e4e3df'
rows = json.load(open(os.path.join(RES, 'sd_conditions.json')))
get = {(r['backbone'], r['dataset'], r['condition']): r for r in rows}
DS = ['FD001', 'FD002', 'FD003', 'FD004']; BB = ['LSTM', 'Transformer']
K = [0.5, 1, 2, 4]
xs = [-k for k in K[::-1]] + [0] + K


def cond(x):
    if x == 0: return 'clean'
    k = abs(x); k = int(k) if k >= 1 else k
    return f'sd_drift{k}' if x > 0 else f'sd_neg{k}'


# ---- Figure 2: trigger errors and coverage against signed drift magnitude
fig, ax = plt.subplots(2, 4, figsize=(7.2, 4.4), sharex=True)
for j, ds in enumerate(DS):
    a, b = ax[0, j], ax[1, j]
    for bb, ls, mk in (('LSTM', '-', 'o'), ('Transformer', '--', 's')):
        e = [get[(bb, ds, cond(x))]['NLL_L20_early'] for x in xs]
        u = [get[(bb, ds, cond(x))]['NLL_L20_late'] for x in xs]
        a.plot(xs, e, ls, color=BLUE, lw=1.4, marker=mk, ms=3.2, mfc='white', mew=1.0)
        a.plot(xs, u, ls, color=ORANGE, lw=1.4, marker=mk, ms=3.2, mfc='white', mew=1.0)
        p = [get[(bb, ds, cond(x))]['picp90_NLL'] for x in xs]
        c = [get[(bb, ds, cond(x))]['cov1_NLL'] for x in xs]
        b.plot(xs, p, ls, color=AQUA, lw=1.4, marker=mk, ms=3.2, mfc='white', mew=1.0)
        b.plot(xs, c, ls, color=VIOLET, lw=1.4, marker=mk, ms=3.2, mfc='white', mew=1.0)
    a.set_title(ds, fontsize=8.5)
    a.set_ylim(-2, 80); b.set_ylim(-0.03, 1.05)
    for z in (a, b):
        z.axvline(0, color=GRID, lw=0.8, zorder=0); z.grid(axis='y', color=GRID, lw=0.5); z.set_axisbelow(True)
    b.axhline(0.90, color='#a3a29b', lw=0.6, ls=':'); b.axhline(0.95, color='#a3a29b', lw=0.6, ls=':')
    b.set_xticks([-4, -2, -1, 0, 1, 2, 4])
    if j == 0:
        a.set_ylabel('% of evaluated engines'); b.set_ylabel('coverage')
    else:
        a.set_yticklabels([]); b.set_yticklabels([])
ax[0, 0].text(2.2, 55, 'premature', color='#0b0b0b', fontsize=7)
ax[0, 0].text(-4.2, 26, 'unrecognised', color='#0b0b0b', fontsize=7)
ax[1, 0].text(-4.2, 0.98, 'one-sided', color='#0b0b0b', fontsize=7)
ax[1, 0].text(0.6, 0.45, 'central 90%', color='#0b0b0b', fontsize=7)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=BLUE, lw=1.4), Line2D([], [], color=ORANGE, lw=1.4),
     Line2D([], [], color=AQUA, lw=1.4), Line2D([], [], color=VIOLET, lw=1.4),
     Line2D([], [], color='#52514e', lw=1.2, marker='o', mfc='white', ms=3.2),
     Line2D([], [], color='#52514e', lw=1.2, ls='--', marker='s', mfc='white', ms=3.2)]
fig.supxlabel('signed ramp end value (within-condition SD; negative = shift downwards)', fontsize=8, y=0.10)
fig.legend(h, ['premature trigger', 'unrecognised need', 'central 90% coverage (PICP)',
               'one-sided coverage of lower bound', 'LSTM', 'Transformer'],
           loc='lower center', ncol=3, frameon=False, fontsize=7, handlelength=2.4, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.11, 1, 1))
fig.savefig(os.path.join(OUT, 'fig2.pdf')); fig.savefig(os.path.join(OUT, 'fig2.png'), dpi=200)

# ---- Figure 3: single-channel ramps (+2 SD), signed error
pc = json.load(open(PC))
names = {'s_2': 'T24', 's_3': 'T30', 's_4': 'T50', 's_6': 'P15', 's_7': 'P30', 's_8': 'Nf', 's_9': 'Nc', 's_11': 'Ps30',
         's_12': 'phi', 's_13': 'NRf', 's_14': 'NRc', 's_15': 'BPR', 's_17': 'htBleed', 's_20': 'W31', 's_21': 'W32'}
chs = [c for c in names if any(r['channel'] == c for r in pc)]
cols = [(bb, ds) for ds in DS for bb in BB]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
cmap = LinearSegmentedColormap.from_list('div', ['#d03b2f', '#f0efec', BLUE])
for k, sign in enumerate((1, -1)):
    M = np.full((len(chs), len(cols)), np.nan)
    for i, c in enumerate(chs):
        for j, (bb, ds) in enumerate(cols):
            r = [x for x in pc if x['channel'] == c and x['backbone'] == bb and x['dataset'] == ds and x['sign'] == sign]
            if r: M[i, j] = r[0]['premature_L20'] - r[0]['unrecognised_L20']
    im = axs[k].imshow(M, cmap=cmap, vmin=-60, vmax=60, aspect='auto')
    axs[k].set_xticks(range(len(cols)))
    axs[k].set_xticklabels([f"{ds[-1]}{'L' if bb == 'LSTM' else 'T'}" for bb, ds in cols], fontsize=7)
    axs[k].set_xlabel('FD00x sub-dataset, L = LSTM, T = Transformer', fontsize=7)
    axs[k].set_title('+2 SD ramp on one channel' if sign == 1 else '$-$2 SD ramp on one channel', fontsize=8)
    for i in range(len(chs)):
        for j in range(len(cols)):
            v = M[i, j]
            if np.isnan(v): axs[k].text(j, i, '·', ha='center', va='center', color='#a3a29b', fontsize=6)
            elif abs(v) >= 10: axs[k].text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=5.6,
                                          color='white' if abs(v) > 35 else '#0b0b0b')
    for s in axs[k].spines.values(): s.set_visible(False)
axs[0].set_yticks(range(len(chs))); axs[0].set_yticklabels([names[c] for c in chs], fontsize=7)
cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.02)
cb.set_label('premature minus unrecognised (% of engines, L = 20)', fontsize=7); cb.ax.tick_params(labelsize=6.5)
fig.savefig(os.path.join(OUT, 'fig3.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(OUT, 'fig3.png'), dpi=200, bbox_inches='tight')
print('ok')
