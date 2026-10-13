"""JRR branch: derived statistics from the per-engine re-derivation and from
released result files. Exploratory (not pre-registered); every output is
written under results_jrr/derived/ and never overwrites results/.
Inputs : results_jrr/derived/per_engine/*.pkl  (jrr_decision_per_engine.py),
         results/degradation/*.json (released).
Outputs: decision_engine_bootstrap.json, output_swap.json,
         label_instant_sensitivity.json, rank_stability.json,
         agreement_with_release.json, foob_coverage_points.csv,
         foob_matched_interpolation.json
"""
import os, json, pickle, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
D = os.path.join(PROJ, 'results_jrr', 'derived'); RES = os.path.join(PROJ, 'results')
BB = ['LSTM', 'Transformer']; DS = ['FD001', 'FD002', 'FD003', 'FD004']
METH = ['NLL', 'MSE_fixed', 'MC_Dropout_fixed', 'Deep_Ensemble', 'CP_norm']
CONDS = ['clean', 'gaussian1pct', 'drift5pct', 'bias5pct']; LS = [10, 20, 30]; RATIOS = [5, 20, 100]
B = 2000; Z = 1.645
PE = {(b, d): pickle.load(open(os.path.join(D, 'per_engine', f'{b}_{d}.pkl'), 'rb')) for b in BB for d in DS}

def eng_vectors(bb, ds, cond, m, L, y=None, ycut=None):
    """per-engine mean (over model/trial cells) indicators"""
    c = PE[(bb, ds)]['cond'][cond]; lb = c['lb'][m]; yy = c['y'] if y is None else y
    trig = lb <= L; ar = yy[None, :] <= L
    unrec = (ar & ~trig).mean(0); prem = (trig & (yy[None, :] > L + 20)).mean(0)
    return unrec, prem, ar[0].astype(float)

def boot_idx(n, rng): return rng.randint(0, n, (B, n))

# ---- D1 engine-level bootstrap of decision rates (paired over engines) -------------------------
rng = np.random.RandomState(20261012)
out1 = {}
for bb in BB:
    for ds in DS:
        n = len(PE[(bb, ds)]['cond']['clean']['y']); idx = boot_idx(n, rng)
        for cond in CONDS:
            for m in METH:
                for L in LS:
                    u, p, a = eng_vectors(bb, ds, cond, m, L)
                    c = PE[(bb, ds)]['cond'][cond]
                    pm = c['lb'][m] <= L; prem_mask = pm & (c['y'][None, :] > L + 20)
                    yu = c['y_unc']
                    sum_pu = (prem_mask * yu[None, :]).mean(0)          # per-engine mean of prem*y_unc
                    r = {}
                    def ci(v): return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                    bu = u[idx].mean(1); bp = p[idx].mean(1)
                    r['unrec'] = [float(u.mean())] + ci(bu); r['prem'] = [float(p.mean())] + ci(bp)
                    cu = u[idx].sum(1) / np.maximum(a[idx].sum(1), 1e-9)
                    r['unrec_cond'] = [float(u.sum() / a.sum()) if a.sum() > 0 else None] + (ci(cu) if a.sum() > 0 else [None, None])
                    if p.sum() > 0:
                        bm = sum_pu[idx].sum(1) / np.maximum(p[idx].sum(1), 1e-12)
                        r['rul_unc_at_trigger'] = [float(sum_pu.sum() / p.sum())] + ci(bm)
                    # across-model (seed) range of the engine-mean rate: NLL / single-model mechanisms only
                    lb = c['lb'][m]
                    if m != 'Deep_Ensemble':
                        nt = lb.shape[0] // 5; seedwise = []
                        for s in range(5):
                            cells = [t * 5 + s for t in range(nt)]
                            tr = lb[cells] <= L
                            seedwise.append(float((tr & (c['y'][None, :] > L + 20)).mean()))
                        r['prem_by_model'] = seedwise
                        seedwise_u = []
                        for s in range(5):
                            cells = [t * 5 + s for t in range(nt)]
                            seedwise_u.append(float(((c['y'][None, :] <= L) & ~(lb[cells] <= L)).mean()))
                        r['unrec_by_model'] = seedwise_u
                    r['n_engines'] = int(n); r['n_at_risk_engines'] = int(a.sum())
                    out1[f'{bb}|{ds}|{cond}|{m}|L{L}'] = r
json.dump(out1, open(os.path.join(D, 'decision_engine_bootstrap.json'), 'w'), indent=1)

# ---- D2 rank stability across cost ratios (paired engine bootstrap) -----------------------------
out2 = {}
for bb in BB:
    for ds in DS:
        n = len(PE[(bb, ds)]['cond']['clean']['y']); idx = boot_idx(n, rng)
        for cond in CONDS:
            for L in LS:
                U = np.stack([eng_vectors(bb, ds, cond, m, L)[0] for m in METH]); P = np.stack([eng_vectors(bb, ds, cond, m, L)[1] for m in METH])
                cost = {r: r * U + P for r in RATIOS}                       # (mech, engine)
                pt = {r: cost[r].mean(1) for r in RATIOS}
                order = {r: tuple(np.argsort(pt[r], kind='stable')) for r in RATIOS}
                rev = len(set(order.values())) > 1; topchg = len({o[0] for o in order.values()}) > 1
                bc = {r: cost[r][:, idx].mean(2) for r in RATIOS}            # (mech, B)
                top = {r: bc[r].argmin(0) for r in RATIOS}
                p_top_change_5_100 = float((top[5] != top[100]).mean())
                # mechanisms practically tied at the minimum (all-zero or identical cost) -> flag
                spread = {r: float(pt[r].max() - pt[r].min()) for r in RATIOS}
                out2[f'{bb}|{ds}|{cond}|L{L}'] = {'reversal': bool(rev), 'top_change': bool(topchg),
                    'p_boot_top5_ne_top100': p_top_change_5_100, 'cost_spread': spread,
                    'top_by_ratio': {str(r): METH[order[r][0]] for r in RATIOS}}
json.dump(out2, open(os.path.join(D, 'rank_stability.json'), 'w'), indent=1)

# ---- D3 mean/scale output swap at the decision condition (heteroscedastic head) -------------------
out3 = {}
for bb in BB:
    for ds in DS:
        c0 = PE[(bb, ds)]['cond']['clean']; mu0 = c0['mu'][0]; s0 = c0['sigma'][0]    # (seeds, eng)
        y = c0['y']; n = len(y); idx = boot_idx(n, rng)
        for cond in ['drift5pct', 'bias5pct', 'gaussian1pct']:
            c1 = PE[(bb, ds)]['cond'][cond]
            for L in LS:
                def rates(mu, sg):
                    lb = mu - Z * sg; tr = lb <= L
                    prem = (tr & (y[None, None, :] > L + 20)).mean((0, 1)); unr = ((y[None, None, :] <= L) & ~tr).mean((0, 1))
                    return prem, unr    # per-engine
                mu1 = c1['mu']; s1 = c1['sigma']                           # (trials, seeds, eng)
                R = {'00': rates(mu0[None], s0[None]), '10': rates(mu1, s0[None] * np.ones_like(mu1)),
                     '01': rates(mu0[None] * np.ones_like(mu1), s1), '11': rates(mu1, s1)}
                res = {}
                for kind, k in [('prem', 0), ('unrec', 1)]:
                    v = {q: R[q][k] for q in R}
                    def stat(w): return float(w.mean()), [float(np.percentile(w[idx].mean(1), 2.5)), float(np.percentile(w[idx].mean(1), 97.5))]
                    res[kind] = {'C00_clean': stat(v['00']), 'mu_perturbed_sigma_clean': stat(v['10']),
                                 'mu_clean_sigma_perturbed': stat(v['01']), 'both_perturbed': stat(v['11']),
                                 'delta_mu_given_sigma_clean': stat(v['10'] - v['00']), 'delta_sigma_given_mu_clean': stat(v['01'] - v['00']),
                                 'delta_mu_given_sigma_pert': stat(v['11'] - v['01']), 'delta_sigma_given_mu_pert': stat(v['11'] - v['10'])}
                res['mean_mu_shift'] = float((mu1 - mu0[None]).mean()); res['mean_sigma_ratio'] = float((s1 / s0[None]).mean())
                out3[f'{bb}|{ds}|{cond}|L{L}'] = res
json.dump(out3, open(os.path.join(D, 'output_swap.json'), 'w'), indent=1)

# ---- D4 label-instant sensitivity ---------------------------------------------------------------
out4 = {}
mx = 0
for bb in BB:
    for ds in DS:
        for cond in CONDS:
            for m in METH:
                for L in LS:
                    c = PE[(bb, ds)]['cond'][cond]; lb = c['lb'][m]; y = c['y']
                    ycur = np.minimum(c['y_unc'] + 1, 125)               # official current-cycle label, 125-clipped
                    tr = lb <= L
                    a = ((y[None] <= L) & ~tr).mean(); b = ((ycur[None] <= L) & ~tr).mean()
                    pa = (tr & (y[None] > L + 20)).mean(); pb = (tr & (ycur[None] > L + 20)).mean()
                    out4[f'{bb}|{ds}|{cond}|{m}|L{L}'] = {'unrec_main': float(a), 'unrec_cur': float(b), 'prem_main': float(pa), 'prem_cur': float(pb)}
                    mx = max(mx, abs(a - b), abs(pa - pb))
out4['_max_abs_change'] = float(mx)
json.dump(out4, open(os.path.join(D, 'label_instant_sensitivity.json'), 'w'), indent=1)

# ---- D5 agreement with released aggregates ----------------------------------------------------------
rel = json.load(open(os.path.join(RES, 'decision', 'maintenance_decision_one_sided.json')))
v2 = json.load(open(os.path.join(RES, 'decision', 'maintenance_decision_rul_corrected_v2.json')))
mxd = 0; n = 0
for bb in BB:
    for ds in DS:
        for cond in CONDS:
            for m in METH:
                for L in LS:
                    u, p, a = eng_vectors(bb, ds, cond, m, L)
                    r = rel[bb][ds][cond][m][str(L)]
                    mxd = max(mxd, abs(u.mean() - r['overall_unrecognised_rate']), abs(p.mean() - r['premature_rate'])); n += 2
json.dump({'n_values_compared': n, 'max_abs_difference': float(mxd),
           'torch_rederivation_note': 'inference re-run with the torch/cuDNN of this session; see JRR_BASELINE.md'},
          open(os.path.join(D, 'agreement_with_release.json'), 'w'), indent=1)

# ---- D6 f_oob vs coverage points from released sweeps ----------------------------------------------------
def g(x):
    while isinstance(x, dict): x = x.get('grand_mean', list(x.values())[0])
    return float(x)
rows = []
sw = json.load(open(os.path.join(RES, 'degradation', 'degradation_sweep_bias_gain_drift.json')))
for bb in BB:
    for deg in ['bias', 'gain', 'drift']:
        for ds in DS:
            blk = sw[bb][deg][ds]
            for lv in ['0.1', '0.5', '1', '2', '5']:
                fo = g(blk['feat_oob'][lv])
                for m in METH:
                    rows.append([bb, deg, ds, m, float(lv), fo, g(blk[m][lv]['picp'])])
noise_files = {'LSTM': [('lstm_noise_armc.json', ['FD001', 'FD002', 'FD004']), ('lstm_fd003_noise_armc.json', ['FD003'])],
               'Transformer': [('transformer_noise_armc.json', DS)]}
for bb, lst in noise_files.items():
    for f, dss in lst:
        nz = json.load(open(os.path.join(RES, 'degradation', f)))
        for ds in dss:
            if ds not in nz: continue
            for lv in ['0.1', '0.5', '1', '2', '5']:
                fo = g(nz[ds]['feat_oob'][lv])
                for m in METH:
                    rows.append([bb, 'noise', ds, m, float(lv), fo, g(nz[ds][m][lv]['picp'])])
with open(os.path.join(D, 'foob_coverage_points.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['backbone', 'degradation', 'dataset', 'mechanism', 'level_pct_range', 'f_oob', 'picp']); w.writerows(rows)
# matched-f_oob interpolation (NLL), f_oob targets within the span of each curve
import collections
curves = collections.defaultdict(list)
for r in rows:
    if r[3] == 'NLL': curves[(r[0], r[2], r[1])].append((r[5], r[6], r[4]))
mi = {}
for (bb, ds, deg), pts in curves.items():
    pts = sorted(pts); xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    d = {}
    for tgt in (0.05, 0.10):
        d[str(tgt)] = float(np.interp(tgt, xs, ys)) if xs.min() <= tgt <= xs.max() else None
    d['f_oob_range'] = [float(xs.min()), float(xs.max())]; d['picp_at_level5'] = float(ys[np.argmax([p[2] for p in pts])])
    mi[f'{bb}|{ds}|{deg}'] = d
json.dump(mi, open(os.path.join(D, 'foob_matched_interpolation.json'), 'w'), indent=1)
print('done', len(rows), 'points; agreement', n, mxd, '; label sensitivity max abs change', mx)
