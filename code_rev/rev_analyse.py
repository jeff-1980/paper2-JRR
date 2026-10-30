"""JRR revision: analyses on the outputs of rev_generate.py.
Run: python rev_analyse.py <outdir_with_pkls> <released_per_engine_dir> <results_dir>
"""
import os, sys, json, pickle, itertools
import numpy as np
from scipy.stats import norm, beta

IN, REL, RES = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(RES, exist_ok=True)
BB = ['LSTM', 'Transformer']; DS = ['FD001', 'FD002', 'FD003', 'FD004']
MECH = ['NLL', 'MSE_fixed', 'MC_Dropout_fixed', 'Deep_Ensemble', 'CP_norm']
NB = 2000
rng0 = np.random.RandomState(20261030)


def conf_q(scores, alpha, w=None):
    """Signed conformal quantile at level 1-alpha; optional weights (normalised)."""
    s = np.asarray(scores)
    if w is None:
        n = len(s); k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
        return float(np.quantile(s, k / n, method='higher'))
    o = np.argsort(s); s = s[o]; w = np.asarray(w, float)[o]; w = w / w.sum()
    # weighted analogue with the (n+1) finite-sample correction expressed in engine units
    cw = np.cumsum(w)
    return float(s[min(np.searchsorted(cw, 1 - alpha), len(s) - 1)])


def cp_q_variants(cal, alpha, test_y=None, R=500, rng=None):
    mu, sg, y, u = cal['mu'], cal['sigma'], cal['y'], cal['unit']
    sc = (mu - y) / np.clip(sg, 1e-6, None)
    out = {'all_windows': conf_q(sc, alpha)}
    uu, inv, cnt = np.unique(u, return_inverse=True, return_counts=True)
    out['engine_weighted'] = conf_q(sc, alpha, w=1.0 / cnt[inv])
    # one window per calibration engine, R draws: list of q
    idx_by_u = [np.where(inv == i)[0] for i in range(len(uu))]
    qs = []
    for _ in range(R):
        pick = np.array([rng.choice(ix) for ix in idx_by_u])
        qs.append(conf_q(sc[pick], alpha))
    out['one_per_engine_draws'] = np.array(qs)
    if test_y is not None:
        # diagnostic: reweight calibration windows so their clipped-label
        # histogram matches the test terminal-label histogram (bins of 10 cycles)
        bins = np.arange(0, 136, 10)
        ht, _ = np.histogram(test_y, bins=bins); hc, _ = np.histogram(y, bins=bins)
        b = np.clip(np.digitize(y, bins) - 1, 0, len(ht) - 1)
        w = np.where(hc[b] > 0, ht[b] / np.maximum(hc[b], 1), 0.0)
        out['label_matched'] = conf_q(sc, alpha, w=w)
    return out


def lower_bounds(c, alpha, cpq):
    """c: condition dict with (trials, seeds, engines) arrays. Returns {mech: (trials*seeds, engines)}."""
    z = norm.ppf(1 - alpha)
    T, S, N = c['mu_nll'].shape
    lb = {'NLL': c['mu_nll'] - z * c['sg_nll'], 'MSE_fixed': c['mu_mse'] - z * c['sg_mse'],
          'MC_Dropout_fixed': c['mu_mc'] - z * c['sg_mc'],
          'CP_norm': c['mu_nll'] - np.array(cpq)[None, :, None] * c['sg_nll']}
    mu_e = c['mu_nll'].mean(1); s2 = (c['sg_nll'] ** 2 + c['mu_nll'] ** 2).mean(1) - mu_e ** 2
    lb['Deep_Ensemble'] = (np.clip(mu_e, 0, 125) - z * np.sqrt(np.clip(s2, 0, None)))[:, None, :]
    out = {m: v.reshape(-1, N) for m, v in lb.items()}
    if np.isnan(out['MC_Dropout_fixed']).any() or c.get('_no_mc'):
        out.pop('MC_Dropout_fixed')
    return out


def per_engine_rates(lb, y, L, tol):
    trig = lb <= L
    late = (~trig) & (y[None, :] <= L)
    early = trig & (y[None, :] > L + tol)
    return late.mean(0), early.mean(0)  # per engine, averaged over trials/models


def boot(v, n=NB, rng=None):
    N = len(v); idx = rng.randint(0, N, (n, N))
    b = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def zero_event_upper(n_eng, conf=0.95):
    """Exact one-sided upper bound when no engine shows the event (Clopper-Pearson)."""
    return float(beta.ppf(conf, 1, n_eng))


recs = {}
for bb in BB:
    for ds in DS:
        f = os.path.join(IN, f'{bb}_{ds}.pkl')
        if os.path.exists(f): recs[(bb, ds)] = pickle.load(open(f, 'rb'))

results = {}
# ---------------------------------------------------------------- 1. one-sided coverage, calibration unit
calib_rows = []
cpq_cache = {}
for (bb, ds), r in recs.items():
    y = r['y']
    for alpha in (0.10, 0.05, 0.01):
        qv = [cp_q_variants(r['calib'][s], alpha, test_y=y, rng=np.random.RandomState(7 + i))
              for i, s in enumerate(r['seeds'])]
        cpq_cache[(bb, ds, alpha)] = qv
    for alpha in (0.05,):
        qv = cpq_cache[(bb, ds, alpha)]
        c = r['cond']['clean']
        base = lower_bounds(c, alpha, [q['all_windows'] for q in qv])
        row = {'backbone': bb, 'dataset': ds, 'alpha': alpha, 'n_engines': len(y)}
        for m in [m_ for m_ in MECH if m_ in base]:
            row[f'cov1_{m}'] = float(np.mean(y[None, :] >= base[m]))
        # central 90% PICP of NLL for reference
        mu, sg = c['mu_nll'][0], c['sg_nll'][0]
        row['picp90_NLL'] = float(np.mean((y >= mu - 1.645 * sg) & (y <= mu + 1.645 * sg)))
        for var in ('engine_weighted', 'label_matched'):
            lbv = lower_bounds(c, alpha, [q[var] for q in qv])['CP_norm']
            row[f'cov1_CP_{var}'] = float(np.mean(y[None, :] >= lbv))
            row[f'q_CP_{var}'] = float(np.mean([q[var] for q in qv]))
        row['q_CP_all_windows'] = float(np.mean([q['all_windows'] for q in qv]))
        # one-per-engine draws: coverage per draw (same draw index across seeds)
        covs = []
        for d in range(len(qv[0]['one_per_engine_draws'])):
            lbv = lower_bounds(c, alpha, [q['one_per_engine_draws'][d] for q in qv])['CP_norm']
            covs.append(np.mean(y[None, :] >= lbv))
        row['cov1_CP_one_per_engine_mean'] = float(np.mean(covs))
        row['cov1_CP_one_per_engine_p05'] = float(np.percentile(covs, 5))
        row['cov1_CP_one_per_engine_p95'] = float(np.percentile(covs, 95))
        row['q_CP_one_per_engine_mean'] = float(np.mean([q['one_per_engine_draws'].mean() for q in qv]))
        row['n_calib_engines'] = int(len(np.unique(r['calib'][r['seeds'][0]]['unit'])))
        row['n_calib_windows'] = int(len(r['calib'][r['seeds'][0]]['y']))
        # engine-bootstrap CI for NLL / CP one-sided coverage
        for m in ('NLL', 'CP_norm'):
            pe = (y[None, :] >= base[m]).mean(0)
            _, lo, hi = boot(pe, rng=rng0)
            row[f'cov1_{m}_lo'] = lo; row[f'cov1_{m}_hi'] = hi
        # decision consequence of the calibration variant under drift and bias at L=20
        for cond in ('clean', 'drift5', 'bias5', 'negdrift5'):
            cc = r['cond'][cond]
            for var in ('all_windows', 'engine_weighted', 'label_matched'):
                lbv = lower_bounds(cc, alpha, [q[var] for q in qv])['CP_norm']
                la, ea = per_engine_rates(lbv, y, 20, 20)
                row[f'CP_{var}_{cond}_late'] = float(la.mean() * 100)
                row[f'CP_{var}_{cond}_early'] = float(ea.mean() * 100)
        calib_rows.append(row)
results['calibration_unit'] = calib_rows

# released CP q check
chk = []
for (bb, ds), r in recs.items():
    rel = pickle.load(open(os.path.join(REL, f'{bb}_{ds}.pkl'), 'rb'))
    chk.append(max(abs(rel['cp_q_lower'][s] - q['all_windows'])
                   for s, q in zip(r['seeds'], cpq_cache[(bb, ds, 0.05)])))
results['cp_q_max_abs_diff_vs_release'] = float(max(chk))

# ---------------------------------------------------------------- 2. condition table (L=20, alpha .05, tol 20)
COND_ORDER = ['clean', 'gauss1', 'bias5', 'drift1', 'drift2', 'drift5', 'negdrift5', 'k1drift5', 'k3drift5',
              'k5drift5', 'grp_temp', 'grp_press', 'grp_speed', 'grp_flowbleed', 'aligned5', 'anti5', 'cont5',
              'rate005', 'rate005_aligned']
cond_rows = []
for (bb, ds), r in recs.items():
    y = r['y']; qv = cpq_cache[(bb, ds, 0.05)]
    mu_clean = r['cond']['clean']['mu_nll'][0]
    for cond in COND_ORDER:
        c = dict(r['cond'][cond]); c['_no_mc'] = cond not in ('clean', 'gauss1', 'bias5', 'drift5')
        lb = lower_bounds(c, 0.05, [q['all_windows'] for q in qv])
        mu, sg = c['mu_nll'], c['sg_nll']
        picp = float(np.mean((y >= mu - 1.645 * sg) & (y <= mu + 1.645 * sg)))
        row = {'backbone': bb, 'dataset': ds, 'condition': cond, 'picp90_NLL': picp,
               'cov1_NLL': float(np.mean(y[None, :] >= lb['NLL'])),
               'foob': float(c['foob'].mean()), 'dmu_NLL': float((mu - mu_clean[None]).mean()),
               'dlogsig_NLL': float(np.log(sg / r['cond']['clean']['sg_nll'][0][None]).mean())}
        for m in [m_ for m_ in MECH if m_ in lb]:
            la, ea = per_engine_rates(lb[m], y, 20, 20)
            row[f'{m}_late'] = float(la.mean() * 100); row[f'{m}_early'] = float(ea.mean() * 100)
            if m in ('NLL', 'CP_norm'):
                _, lo, hi = boot(ea * 100, rng=rng0); row[f'{m}_early_lo'] = lo; row[f'{m}_early_hi'] = hi
                _, lo, hi = boot(la * 100, rng=rng0); row[f'{m}_late_lo'] = lo; row[f'{m}_late_hi'] = hi
                ne = int((la > 0).sum()); row[f'{m}_late_engines_any'] = ne
                row[f'{m}_late_zero_upper95'] = zero_event_upper(len(y)) * 100 if ne == 0 else None
        # subset spread for k-channel conditions (trial = subset)
        if c['mu_nll'].shape[0] > 1 and cond.startswith('k'):
            per_trial = []
            for t in range(c['mu_nll'].shape[0]):
                lbt = c['mu_nll'][t] - 1.645 * c['sg_nll'][t]
                per_trial.append(np.mean((lbt <= 20) & (y[None, :] > 40)) * 100)
            row['NLL_early_subset_min'] = float(min(per_trial)); row['NLL_early_subset_max'] = float(max(per_trial))
        cond_rows.append(row)
results['conditions'] = cond_rows

# ---------------------------------------------------------------- 3. decision sensitivity grid
grid = []
for (bb, ds), r in recs.items():
    y = r['y']
    for alpha, tol, L in itertools.product((0.10, 0.05, 0.01), (10, 20, 40), (10, 20, 30)):
        qv = cpq_cache[(bb, ds, alpha)]
        for cond in ('clean', 'gauss1', 'bias5', 'drift5', 'negdrift5', 'aligned5', 'anti5', 'k3drift5', 'rate005'):
            c = dict(r['cond'][cond]); c['_no_mc'] = cond not in ('clean', 'gauss1', 'bias5', 'drift5')
            lb = lower_bounds(c, alpha, [q['all_windows'] for q in qv])
            row = {'backbone': bb, 'dataset': ds, 'alpha': alpha, 'tol': tol, 'L': L, 'condition': cond}
            for m in [m_ for m_ in MECH if m_ in lb]:
                la, ea = per_engine_rates(lb[m], y, L, tol)
                row[f'{m}_late'] = float(la.mean() * 100); row[f'{m}_early'] = float(ea.mean() * 100)
            grid.append(row)
results['sensitivity_grid'] = grid

# rank changes across cost ratios for each alpha and tolerance band, plus the regret of
# choosing the mechanism under a wrong cost ratio (cost points at the true ratio)
rank = []; regret_rows = []
for alpha in (0.10, 0.05, 0.01):
    for tol in (10, 20, 40):
        nrev = ntop = ntot = 0; regrets = []
        for row in grid:
            if row['alpha'] != alpha or row['tol'] != tol or row['condition'] not in ('clean', 'gauss1', 'bias5', 'drift5'):
                continue
            cost = {cr: np.array([cr * row[f'{m}_late'] + row[f'{m}_early'] for m in MECH]) for cr in (5, 20, 100)}
            ranks = {cr: tuple(np.argsort(cost[cr], kind='stable')) for cr in cost}
            ntot += 1; nrev += len(set(ranks.values())) > 1; ntop += len(set(rk[0] for rk in ranks.values())) > 1
            reg = max(cost[t][ranks[a][0]] - cost[t].min() for a in cost for t in cost)
            # regret relative to the cost of the best mechanism at the true ratio
            regrets.append(reg)
            if alpha == 0.05 and tol == 20:
                regret_rows.append({'backbone': row['backbone'], 'dataset': row['dataset'], 'condition': row['condition'],
                                    'L': row['L'], 'max_regret_cost_points': float(reg),
                                    'top': {cr: MECH[ranks[cr][0]] for cr in cost}})
        regrets = np.array(regrets)
        rank.append({'alpha': alpha, 'tol': tol, 'groups': ntot, 'ranking_changes': nrev, 'top_changes': ntop,
                     'regret_gt0': int((regrets > 1e-9).sum()), 'regret_gt1pt': int((regrets > 1).sum()),
                     'regret_gt5pt': int((regrets > 5).sum()), 'regret_max': float(regrets.max())})
results['regret_alpha05_tol20'] = regret_rows
results['rank_changes_by_alpha_tol'] = rank

# ---------------------------------------------------------------- 4. trajectory first trigger (NLL head)
traj_rows = []
for (bb, ds), r in recs.items():
    meta = r['traj_meta']; u = meta['unit']; yu = meta['y_unc']
    units = np.unique(u)
    for tn, tr in r['traj'].items():
        lb_all = tr['mu'] - 1.645 * tr['sigma']   # (seeds, windows)
        for L in (10, 20, 30):
            early = []; timely = []; never = []; missed = []; atrisk = []; forgone = []
            for s in range(lb_all.shape[0]):
                e = t = nv = ms = ar = 0; fg = []
                for un in units:
                    ix = np.where(u == un)[0]
                    trig = np.where(lb_all[s, ix] <= L)[0]
                    reach = yu[ix[-1]] <= L
                    ar += reach
                    if len(trig) == 0:
                        nv += 1; ms += reach
                    else:
                        rul_at = yu[ix[trig[0]]]
                        if rul_at > L + 20:
                            e += 1; fg.append(rul_at - L)
                        else:
                            t += 1
                n = len(units)
                early.append(e / n * 100); timely.append(t / n * 100); never.append(nv / n * 100)
                missed.append(ms / max(ar, 1) * 100); atrisk.append(ar); forgone.append(np.mean(fg) if fg else np.nan)
            traj_rows.append({'backbone': bb, 'dataset': ds, 'scenario': tn, 'L': L, 'n_engines': len(units),
                              'first_trigger_premature_pct': float(np.mean(early)),
                              'first_trigger_timely_pct': float(np.mean(timely)),
                              'never_triggered_pct': float(np.mean(never)),
                              'engines_reaching_L': float(np.mean(atrisk)),
                              'missed_among_reaching_L_pct': float(np.mean(missed)),
                              'mean_forgone_cycles_premature': float(np.nanmean(forgone)) if not np.all(np.isnan(forgone)) else None})
results['trajectory_first_trigger'] = traj_rows

json.dump(results, open(os.path.join(RES, 'revision_results.json'), 'w'), indent=1, default=float)
import csv
for key in ('calibration_unit', 'conditions', 'sensitivity_grid', 'trajectory_first_trigger'):
    rows = results[key]
    with open(os.path.join(RES, f'{key}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r_ in rows for k in r_))); w.writeheader(); w.writerows(rows)
print('done; cp q diff vs release', results['cp_q_max_abs_diff_vs_release'])
