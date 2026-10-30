"""JRR revision: summaries of the within-condition-SD perturbations (rev_generate_sd.py outputs).
Run: python rev_analyse_sd.py <out_sd dir> <results dir>
"""
import os, sys, json, csv
import numpy as np
from scipy.stats import norm

IN, RES = sys.argv[1], sys.argv[2]
os.makedirs(RES, exist_ok=True)
MECH = ['NLL', 'MSE_fixed', 'MC_Dropout_fixed', 'Deep_Ensemble', 'CP_norm']
rng = np.random.RandomState(20261031)


def conf_q(scores, alpha):
    n = len(scores); k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.quantile(scores, k / n, method='higher'))


def lbs(c, cpq, alpha=0.05):
    z = norm.ppf(1 - alpha); N = c['mu_nll'].shape[2]
    lb = {'NLL': c['mu_nll'] - z * c['sg_nll'], 'MSE_fixed': c['mu_mse'] - z * c['sg_mse'],
          'MC_Dropout_fixed': c['mu_mc'] - z * c['sg_mc'],
          'CP_norm': c['mu_nll'] - np.array(cpq)[None, :, None] * c['sg_nll']}
    mu_e = c['mu_nll'].mean(1); s2 = (c['sg_nll'] ** 2 + c['mu_nll'] ** 2).mean(1) - mu_e ** 2
    lb['Deep_Ensemble'] = (np.clip(mu_e, 0, 125) - z * np.sqrt(np.clip(s2, 0, None)))[:, None, :]
    return {m: v.reshape(-1, N) for m, v in lb.items()}


def boot(v):
    idx = rng.randint(0, len(v), (2000, len(v))); b = v[idx].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


rows = []
import pickle
for bb in ['LSTM', 'Transformer']:
    for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
        f = os.path.join(IN, f'{bb}_{ds}.pkl')
        if not os.path.exists(f): continue
        r = pickle.load(open(f, 'rb')); y = r['y']
        cpq = []
        for s in r['seeds']:
            cal = r['calib'][s]; cpq.append(conf_q((cal['mu'] - cal['y']) / np.clip(cal['sigma'], 1e-6, None), 0.05))
        mu0 = r['cond']['clean']['mu_nll'][0]
        for cond, c in r['cond'].items():
            lb = lbs(c, cpq)
            mu, sg = c['mu_nll'], c['sg_nll']
            row = {'backbone': bb, 'dataset': ds, 'condition': cond,
                   'picp90_NLL': float(np.mean((y >= mu - 1.645 * sg) & (y <= mu + 1.645 * sg))),
                   'cov1_NLL': float(np.mean(y[None, :] >= lb['NLL'])), 'foob': float(c['foob'].mean()),
                   'dmu_NLL': float((mu - mu0[None]).mean())}
            for L in (10, 20, 30):
                for m in MECH:
                    trig = lb[m] <= L
                    la = ((~trig) & (y[None] <= L)).mean(0) * 100; ea = (trig & (y[None] > L + 20)).mean(0) * 100
                    row[f'{m}_L{L}_late'] = float(la.mean()); row[f'{m}_L{L}_early'] = float(ea.mean())
                    if m == 'NLL' and L == 20:
                        row['NLL_L20_early_lo'], row['NLL_L20_early_hi'] = boot(ea)
                        row['NLL_L20_late_lo'], row['NLL_L20_late_hi'] = boot(la)
            if c['mu_nll'].shape[0] > 1 and '_k' in cond:
                pt = [float(np.mean(((c['mu_nll'][t] - 1.645 * c['sg_nll'][t]) <= 20) & (y[None] > 40)) * 100)
                      for t in range(c['mu_nll'].shape[0])]
                pl = [float(np.mean(((c['mu_nll'][t] - 1.645 * c['sg_nll'][t]) > 20) & (y[None] <= 20)) * 100)
                      for t in range(c['mu_nll'].shape[0])]
                row['NLL_L20_early_subset_min'] = min(pt); row['NLL_L20_early_subset_max'] = max(pt)
                row['NLL_L20_early_subset_median'] = float(np.median(pt))
                row['NLL_L20_late_subset_max'] = max(pl)
            rows.append(row)
json.dump(rows, open(os.path.join(RES, 'sd_conditions.json'), 'w'), indent=1)
with open(os.path.join(RES, 'sd_conditions.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r_ in rows for k in r_))); w.writeheader(); w.writerows(rows)
print('rows', len(rows))

# ---------------- decision sensitivity: bound level x tolerance band x lead time (SD-unit conditions)
import itertools
grid = []; recs = {}
for bb in ['LSTM', 'Transformer']:
    for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
        f = os.path.join(IN, f'{bb}_{ds}.pkl')
        if os.path.exists(f): recs[(bb, ds)] = pickle.load(open(f, 'rb'))
SENS = ('clean', 'sd_drift1', 'sd_drift2', 'sd_neg2', 'sd_bias2', 'sd_aligned2', 'sd_anti2', 'sd_k3drift2')
for (bb, ds), r in recs.items():
    y = r['y']
    for alpha in (0.10, 0.05, 0.01):
        cpq = [conf_q((r['calib'][s]['mu'] - r['calib'][s]['y']) / np.clip(r['calib'][s]['sigma'], 1e-6, None), alpha)
               for s in r['seeds']]
        for cond in SENS:
            lb = lbs(r['cond'][cond], cpq, alpha)
            for tol, L in itertools.product((10, 20, 40), (10, 20, 30)):
                row = {'backbone': bb, 'dataset': ds, 'alpha': alpha, 'tol': tol, 'L': L, 'condition': cond}
                for m in MECH:
                    trig = lb[m] <= L
                    row[f'{m}_late'] = float(((~trig) & (y[None] <= L)).mean() * 100)
                    row[f'{m}_early'] = float((trig & (y[None] > L + tol)).mean() * 100)
                grid.append(row)
with open(os.path.join(RES, 'sd_sensitivity_grid.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(grid[0].keys())); w.writeheader(); w.writerows(grid)
# direction pattern stability: positive drift -> premature > late; negative -> late > premature
summ = []
for alpha in (0.10, 0.05, 0.01):
    for tol in (10, 20, 40):
        sub = [g for g in grid if g['alpha'] == alpha and g['tol'] == tol]
        pos = [g for g in sub if g['condition'] == 'sd_drift2']; neg = [g for g in sub if g['condition'] == 'sd_neg2']
        ok_pos = sum(g[f'{m}_early'] > g[f'{m}_late'] for g in pos for m in MECH)
        ok_neg = sum(g[f'{m}_late'] > g[f'{m}_early'] for g in neg for m in MECH)
        npos = len(pos) * len(MECH)
        prem = [g['NLL_early'] for g in pos if g['L'] == 20]; late = [g['NLL_late'] for g in neg if g['L'] == 20]
        # cost ranking changes and regret over the clean / drift / bias / k3 conditions
        nrev = ntop = ntot = 0; rel_reg = []
        for g in sub:
            if g['condition'] not in ('clean', 'sd_drift1', 'sd_bias2', 'sd_k3drift2'): continue
            cost = {cr: np.array([cr * g[f'{m}_late'] + g[f'{m}_early'] for m in MECH]) for cr in (5, 20, 100)}
            rk = {cr: tuple(np.argsort(cost[cr], kind='stable')) for cr in cost}
            ntot += 1; nrev += len(set(rk.values())) > 1; ntop += len({v[0] for v in rk.values()}) > 1
            rel_reg.append(max((cost[t][rk[a][0]] - cost[t].min()) / max(cost[t].min(), 1.0) for a in cost for t in cost))
        rel_reg = np.array(rel_reg)
        summ.append({'alpha': alpha, 'tol': tol, 'pos_premature_dominant': f'{ok_pos}/{npos}',
                     'neg_late_dominant': f'{ok_neg}/{len(neg) * len(MECH)}',
                     'NLL_L20_premature_sd_drift2_range': [round(min(prem), 1), round(max(prem), 1)],
                     'NLL_L20_late_sd_neg2_range': [round(min(late), 1), round(max(late), 1)],
                     'rank_groups': ntot, 'ranking_changes': nrev, 'top_changes': ntop,
                     'relative_regret_gt10pct': int((rel_reg > 0.10).sum()), 'relative_regret_max': float(rel_reg.max())})
json.dump(summ, open(os.path.join(RES, 'sd_sensitivity_summary.json'), 'w'), indent=1)
for s_ in summ: print(s_)
