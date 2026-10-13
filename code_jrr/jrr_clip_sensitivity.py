"""JRR branch (exploratory, no GPU, no retraining). Predicted-mean clipping sensitivity of the stress-test results.
Reads the stored per-engine NLL means/scales (results_jrr/derived/per_engine/*.pkl) and recomputes, with the
heteroscedastic mean clipped to [0,125] (variant A, as in the v1.8 clean benchmark) instead of left unclipped
(variant U, the stress-test processing of the headline analyses):
  * trigger rates (premature / unrecognised) for NLL, CP_norm and Deep_Ensemble (members clipped individually, then averaged),
    L in {10,20,30}, conditions clean / noise 1% / drift 5% / bias 5%;
  * terminal-window 90% PICP of the heteroscedastic head for the same conditions.
Output: results_jrr/derived/clip_sensitivity.json
"""
import os, sys, json, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
D = os.path.join(PROJ, 'results_jrr', 'derived'); Z = 1.645
BB = ['LSTM', 'Transformer']; DS = ['FD001', 'FD002', 'FD003', 'FD004']; CONDS = ['clean', 'gaussian1pct', 'drift5pct', 'bias5pct']
def rates(lb, y, L):  # lb (..., engines); y (engines,)
    tr = lb <= L
    return float((tr & (y > L + 20)).mean()), float(((y <= L) & ~tr).mean())
out = {'decision': {}, 'picp': {}}
for bb in BB:
    for ds in DS:
        rec = pickle.load(open(os.path.join(D, 'per_engine', f'{bb}_{ds}.pkl'), 'rb'))
        q = np.array([rec['cp_q_lower'][s] for s in rec['seeds']])[None, :, None]
        for c in CONDS:
            r = rec['cond'][c]; mu, sg, y = r['mu'], r['sigma'], r['y']          # (trials, seeds, engines)
            for var, mm in [('U', mu), ('A', np.clip(mu, 0, 125))]:
                me = mm.mean(1); s2 = (sg ** 2 + mm ** 2).mean(1) - me ** 2
                lbs = {'NLL': mm - Z * sg, 'CP_norm': mm - q * sg,
                       'Deep_Ensemble': np.clip(me, 0, 125) - Z * np.sqrt(np.clip(s2, 0, None))}
                for L in (10, 20, 30):
                    for m, lb in lbs.items():
                        pr = [rates(lb[t, s] if lb.ndim == 3 else lb[t], y, L) for t in range(lb.shape[0]) for s in (range(lb.shape[1]) if lb.ndim == 3 else [0])]
                        out['decision'].setdefault(f'{bb}|{ds}|{c}|{m}|L{L}', {})[var] = {'prem': float(np.mean([p[0] for p in pr])), 'unrec': float(np.mean([p[1] for p in pr]))}
                lo, hi = mm - Z * sg, mm + Z * sg
                out['picp'].setdefault(f'{bb}|{ds}|{c}', {})[var] = float(((y >= lo) & (y <= hi)).mean())
dd = [max(abs(v['U'][k] - v['A'][k]) for k in ('prem', 'unrec')) for v in out['decision'].values()]
byk = {}
for k, v in out['decision'].items():
    m = k.split('|')[3]; byk[m] = max(byk.get(m, 0), max(abs(v['U'][x] - v['A'][x]) for x in ('prem', 'unrec')))
out['summary'] = {'max_abs_rate_diff_by_mechanism': byk, 'n_cells': len(dd), 'max_abs_rate_diff_all': float(max(dd))}
pd_ = {c: [abs(v['U'] - v['A']) for k, v in out['picp'].items() if k.endswith('|' + c)] for c in CONDS}
out['summary']['picp_abs_diff_max_by_cond'] = {c: float(max(v)) for c, v in pd_.items()}
json.dump(out, open(os.path.join(D, 'clip_sensitivity.json'), 'w'), indent=1)
print(json.dumps(out['summary'], indent=1))
