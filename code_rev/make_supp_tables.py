"""Supplement S10 tables (signature alignment, offsets, channel subsets, accumulating drift; per-channel ramps).
Run: python make_supp_tables.py <work dir> <out dir>"""
import os, sys, json

W, OUT = sys.argv[1], sys.argv[2]
rows = json.load(open(os.path.join(W, 'res_sd', 'sd_conditions.json')))
get = {(r['backbone'], r['dataset'], r['condition']): r for r in rows}
pc = json.load(open(os.path.join(W, 'perchannel_sd.json')))
DS = ['FD001', 'FD002', 'FD003', 'FD004']; BB = ['LSTM', 'Transformer']
MECH = [('MSE_fixed', 'Fixed var.'), ('NLL', 'Heterosc.'), ('MC_Dropout_fixed', 'MC Drop.'),
        ('Deep_Ensemble', 'Ensemble'), ('CP_norm', 'CP-norm')]
NAMES = {'s_2': 'T24', 's_3': 'T30', 's_4': 'T50', 's_6': 'P15', 's_7': 'P30', 's_8': 'Nf', 's_9': 'Nc', 's_11': 'Ps30',
         's_12': 'phi', 's_13': 'NRf', 's_14': 'NRc', 's_15': 'BPR', 's_17': 'htBleed', 's_20': 'W31', 's_21': 'W32'}


def f(v, d=1):
    return '--' if v is None or v != v else f'{v:.{d}f}'


# Table A: heteroscedastic head, premature / unrecognised at L = 20 by scenario
COND = [('sd_aligned1', 'Aligned 1'), ('sd_aligned2', 'Aligned 2'), ('sd_anti1', 'Anti 1'), ('sd_anti2', 'Anti 2'),
        ('sd_bias1', 'Offset 1'), ('sd_bias2', 'Offset 2'), ('sd_k1drift2', '1 ch.'), ('sd_k3drift2', '3 ch.'),
        ('sd_rate2', 'Rate'), ('sd_rate2_aligned', 'Rate al.')]
L = ['\\begin{tabular}{l' + 'c' * len(COND) + '}\\toprule',
     'Cell & ' + ' & '.join(n for _, n in COND) + ' \\\\ \\midrule']
for bb in BB:
    for ds in DS:
        cells = []
        for c, _ in COND:
            r = get.get((bb, ds, c))
            cells.append('--' if r is None else f"{f(r['NLL_L20_early'], 0)} / {f(r['NLL_L20_late'])}")
        L.append(f'{bb} {ds} & ' + ' & '.join(cells) + ' \\\\')
    if bb == 'LSTM': L.append('\\midrule')
L.append('\\bottomrule\\end{tabular}')
open(os.path.join(OUT, 'supp_table_scenarios.tex'), 'w').write('\n'.join(L) + '\n')

# Table B: channel-subset spread (median [min, max] over ten draws), premature at L = 20
L = ['\\begin{tabular}{lcc}\\toprule',
     'Cell & One channel, $+2$ SD & Three channels, $+2$ SD \\\\ \\midrule']
for bb in BB:
    for ds in DS:
        cells = []
        for c in ('sd_k1drift2', 'sd_k3drift2'):
            r = get[(bb, ds, c)]
            cells.append(f"{f(r['NLL_L20_early_subset_median'])} [{f(r['NLL_L20_early_subset_min'])}, "
                         f"{f(r['NLL_L20_early_subset_max'])}]")
        L.append(f'{bb} {ds} & ' + ' & '.join(cells) + ' \\\\')
L.append('\\bottomrule\\end{tabular}')
open(os.path.join(OUT, 'supp_table_subsets.tex'), 'w').write('\n'.join(L) + '\n')

# Table C: all five mechanisms, aligned / anti-aligned 2 SD, L = 20
L = ['\\begin{tabular}{l' + 'c' * len(MECH) + '}\\toprule',
     'Cell & ' + ' & '.join(n for _, n in MECH) + ' \\\\ \\midrule']
for c, lab in (('sd_aligned2', 'aligned, premature'), ('sd_anti2', 'anti-aligned, unrecognised')):
    L.append(f'\\multicolumn{{{len(MECH) + 1}}}{{l}}{{\\emph{{$2$ SD {lab} (\\%)}}}} \\\\')
    for bb in BB:
        for ds in DS:
            r = get[(bb, ds, c)]
            key = 'early' if c == 'sd_aligned2' else 'late'
            L.append(f'{bb} {ds} & ' + ' & '.join(f(r.get(f'{m}_L20_{key}')) for m, _ in MECH) + ' \\\\')
L.append('\\bottomrule\\end{tabular}')
open(os.path.join(OUT, 'supp_table_mech.tex'), 'w').write('\n'.join(L) + '\n')

# Table D: per-channel ramps, premature and unrecognised at L = 20, both signs
cols = [(bb, ds) for ds in DS for bb in BB]
L = ['\\begin{tabular}{ll' + 'c' * len(cols) + '}\\toprule',
     'Channel & Sign & ' + ' & '.join(f"{ds[-1]}{'L' if bb == 'LSTM' else 'T'}" for bb, ds in cols) + ' \\\\ \\midrule']
for ch, nm in NAMES.items():
    for sign in (1, -1):
        cells = []
        for bb, ds in cols:
            r = [x for x in pc if x['channel'] == ch and x['backbone'] == bb and x['dataset'] == ds and x['sign'] == sign]
            cells.append('$\\cdot$' if not r else f"{r[0]['premature_L20']:.0f}/{r[0]['unrecognised_L20']:.0f}")
        L.append(f"{nm if sign == 1 else ''} & {'$+$' if sign == 1 else '$-$'} & " + ' & '.join(cells) + ' \\\\')
L.append('\\bottomrule\\end{tabular}')
open(os.path.join(OUT, 'supp_table_channels.tex'), 'w').write('\n'.join(L) + '\n')
print('ok')
