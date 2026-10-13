"""JRR branch (exploratory, inference only). Re-evaluates the CLEAN terminal-window benchmark through the same code path
as the stress tests (sweep_engine.infer_* functions) under two predicted-mean post-processing protocols:
  A 'benchmark' (v1.8 eval_*_b1.py / eval_*_deep_ensemble.py): every mechanism's predicted mean is clipped to [0,125]
    (ensemble members clipped individually, then averaged).
  B 'stress-test' (sweep_engine / maintenance_decision scripts): NLL head, CP-norm and ensemble-member means are NOT clipped;
    MC Dropout, fixed-variance and the ensemble mean are clipped to [0,125].
The true-label clip at 125 is a separate matter and is identical in both.
Output: results_jrr/derived/clean_protocol_comparison.json  (per backbone x sub-dataset x mechanism: PICP, MPIW, IS, ECE, RMSE,
mean over the five seeds; MC Dropout also over five dropout-sampling repeats).
"""
import os, sys, json
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, 'code'))
import common as C, noise_injection as V4, transformer_common as T2, sweep_engine as E
ALPHA = 0.10; Z = 1.645; LV = E.CONF_LEVELS
METH = ['MSE_fixed', 'NLL', 'MC_Dropout_fixed', 'Deep_Ensemble', 'CP_norm']
def iscore(y, lo, hi): return float(np.mean((hi - lo) + (2 / ALPHA) * np.maximum(lo - y, 0) + (2 / ALPHA) * np.maximum(y - hi, 0)))
def met(y, mu, sg, z=Z, q_by_level=None):
    lo, hi = mu - z * sg, mu + z * sg
    if q_by_level is None: ece = C.compute_ece(mu, sg, y, LV)
    else:
        emp = [np.mean((y >= mu - q_by_level[f"{p:.2f}"] * sg) & (y <= mu + q_by_level[f"{p:.2f}"] * sg)) for p in LV]
        ece = float(np.mean(np.abs(np.array(emp) - LV)))
    rmse, _ = C.rmse_score(y, mu)
    return {'picp': float(np.mean((y >= lo) & (y <= hi))), 'mpiw': float(np.mean(hi - lo)), 'is': iscore(y, lo, hi), 'ece': float(ece), 'rmse': float(rmse)}
def load_cp(ds, bb):
    # released split-CP result files carry the normalised-score quantile q and q_by_level used by the sweeps
    f = os.path.join(PROJ, 'results', 'clean', {'LSTM': 'lstm_split_cp_per_seed.json', 'Transformer': 'transformer_split_cp_per_seed.json'}[bb])
    if bb == 'LSTM' and ds == 'FD003': f = os.path.join(PROJ, 'results', 'clean', 'lstm_fd003_split_cp_per_seed.json')
    d = json.load(open(f)); lst = d if isinstance(d, list) else d[ds]
    return {str(r['seed']): r for r in lst}
def avg(lst): return {k: float(np.mean([m[k] for m in lst])) for k in lst[0]}
if __name__ == '__main__':
    C.require_fixed_hashseed(); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = {}
    for bb in ['LSTM', 'Transformer']:
        for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
            tr_raw, test_raw, true_ruls, fc, _ = V4.load_raw_train_test_and_scaler(ds)
            Xr, y, _ = V4.extract_raw_windows(test_raw, fc, true_ruls, mode='test')
            canon = E.CANON
            models = {s: E.load_models_for_seed(ds, bb, s, dev) for s in C.SEEDS}
            avar = {s: E.calib_aleatory_var(ds, bb, s, dev, mc_model=models[s][1]) for s in C.SEEDS}
            cp_by_seed = load_cp(ds, bb)
            res = {'A': {m: [] for m in METH}, 'B': {m: [] for m in METH}}; mem = {'A': [], 'B': []}
            for s in C.SEEDS:
                _, _, _, _, sc = V4.load_raw_train_test_and_scaler_leakfree(ds, canon[ds][str(s)]['fit_units'])
                X = torch.tensor(V4.scale_raw_windows(Xr, sc), dtype=torch.float32).to(dev)
                nll, mc, cp = models[s]
                mu, ls = E.infer_nll(nll, X); sg = np.exp(ls) * 125.0
                muc, lsc = E.infer_nll(cp, X); sgc = np.exp(lsc) * 125.0
                q = cp_by_seed[str(s)]['cp_norm']['q']; qbl = cp_by_seed[str(s)]['cp_norm']['q_by_level']
                for var, clipf in [('A', lambda a: np.clip(a, 0, 125)), ('B', lambda a: a)]:
                    res[var]['NLL'].append(met(y, clipf(mu), sg))
                    res[var]['CP_norm'].append(met(y, clipf(muc), sgc, z=q, q_by_level=qbl))
                    mem[var].append((clipf(mu), sg))
                reps = []
                for t in range(E.N_TRIALS):
                    mu_m, sg_m = E.infer_mc_dropout(mc, X, T=50, aleatory_var=avar[s], seed=C.stable_seed(ds, bb, 'clean0', t, s, 'mc_dropout')); reps.append(met(y, mu_m, sg_m))
                for var in 'AB': res[var]['MC_Dropout_fixed'].append(avg(reps))
                mu_f, sg_f = E.infer_mse_fixed(mc, X, sigma_fixed=float(np.sqrt(avar[s])))
                for var in 'AB': res[var]['MSE_fixed'].append(met(y, mu_f, sg_f))
            for var in 'AB':
                mm = np.stack([m for m, _ in mem[var]]); ss = np.stack([s_ for _, s_ in mem[var]])
                me = mm.mean(0); s2 = (ss ** 2 + mm ** 2).mean(0) - me ** 2
                res[var]['Deep_Ensemble'].append(met(y, np.clip(me, 0, 125), np.sqrt(np.clip(s2, 0, None))))
            out[f'{bb}|{ds}'] = {var: {m: avg(v) for m, v in res[var].items()} for var in 'AB'}
            print(bb, ds, 'NLL picp A %.4f B %.4f' % (out[f'{bb}|{ds}']['A']['NLL']['picp'], out[f'{bb}|{ds}']['B']['NLL']['picp']), flush=True)
    json.dump(out, open(os.path.join(PROJ, 'results_jrr', 'derived', 'clean_protocol_comparison.json'), 'w'), indent=1)
