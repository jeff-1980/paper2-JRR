"""JRR branch, exploratory re-derivation (NOT part of the v1.8 pipeline).

Re-runs the inference of maintenance_decision_one_sided.py for the decision
stress test but KEEPS per-engine lower bounds (and NLL mu/sigma), which the
v1.8 release only stored as aggregated rates.  Purpose:
  (1) independent re-check of results/decision/*.json (aggregate agreement);
  (2) engine-level resampling (paired across mechanisms/conditions);
  (3) same-condition mean/scale output swap at the decision point;
  (4) label-instant sensitivity (next-cycle label vs official current-cycle).
Writes only to results_jrr/derived/. Inference only, no training.
Run: PYTHONHASHSEED=0 RUL_DATA_DIR=<cmapss dir> python code_jrr/jrr_decision_per_engine.py
"""
import os, sys, json, pickle
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'code'))
import common as C
import noise_injection as V4
import transformer_common as T2
import sweep_engine as E
from maintenance_decision_one_sided import onesided_cp_q
from maintenance_decision_two_sided import calib_sigma_fixed, get_raw_test_window

PROJ = os.path.dirname(HERE)
OUT = os.path.join(PROJ, 'results_jrr', 'derived', 'per_engine')
os.makedirs(OUT, exist_ok=True)
DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']; BACKBONES = ['LSTM', 'Transformer']
CONDITIONS = ['clean', 'gaussian1pct', 'drift5pct', 'bias5pct']
METHODS = ['NLL', 'MSE_fixed', 'MC_Dropout_fixed', 'Deep_Ensemble', 'CP_norm']

if __name__ == '__main__':
    C.require_fixed_hashseed()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    canon = json.load(open(os.path.join(PROJ, 'results', 'canonical_splits.json')))
    only = sys.argv[1:]  # optional "LSTM/FD002" filters
    for bb in BACKBONES:
        for ds in DATASETS:
            tag = f'{bb}_{ds}'
            if only and f'{bb}/{ds}' not in only: continue
            if os.path.exists(os.path.join(OUT, tag + '.pkl')): continue
            print('==', tag, flush=True)
            tr_raw, _, _, fc, _ = V4.load_raw_train_test_and_scaler(ds)
            full_scale = V4.sensor_only_scale(fc, V4.fit_fullscale_range(tr_raw, fc))
            scal, avar, mcm, nllm, cpq = {}, {}, {}, {}, {}
            for s in C.SEEDS:
                fu = canon[ds][str(s)]['fit_units']
                _, _, _, _, scal[s] = V4.load_raw_train_test_and_scaler_leakfree(ds, fu)
                avar[s], mcm[s] = calib_sigma_fixed(ds, bb, s, canon, dev)
                nllm[s] = T2.load_checkpoint_model_t2(bb, T2.nll_ckpt_path(bb, ds, s), dev)
                cpq[s] = onesided_cp_q(ds, bb, s, canon, dev)
            rec = {'seeds': list(C.SEEDS), 'cp_q_lower': cpq, 'cond': {}}
            for cond in CONDITIONS:
                trials = range(V4.N_TRIALS) if cond in ('gaussian1pct', 'bias5pct') else [0]
                lbs = {m: [] for m in METHODS}; mu_all = []; sg_all = []; ys = []
                for t in trials:
                    X_raw, y_ref, _, test_raw, true_ruls = get_raw_test_window(ds, cond, t, full_scale, None)
                    ys.append(y_ref)
                    mu_t = []; sg_t = []
                    for s in C.SEEDS:
                        Xs = torch.tensor(V4.scale_raw_windows(X_raw, scal[s]), dtype=torch.float32).to(dev)
                        mu_n, ls_n = E.infer_nll(nllm[s], Xs); sg_n = np.exp(ls_n) * 125.0
                        mu_mc, sg_mc = E.infer_mc_dropout(mcm[s], Xs, T=50, aleatory_var=avar[s],
                                          seed=C.stable_seed(ds, bb, cond, t, s, 'mc_dropout'))
                        mu_ms, sg_ms = E.infer_mse_fixed(mcm[s], Xs, sigma_fixed=float(np.sqrt(avar[s])))
                        lbs['NLL'].append(mu_n - C.Z_SCORE * sg_n)
                        lbs['MSE_fixed'].append(mu_ms - C.Z_SCORE * sg_ms)
                        lbs['MC_Dropout_fixed'].append(mu_mc - C.Z_SCORE * sg_mc)
                        lbs['CP_norm'].append(mu_n - cpq[s] * sg_n)
                        mu_t.append(mu_n); sg_t.append(sg_n)
                    mu_t = np.stack(mu_t); sg_t = np.stack(sg_t)
                    mu_all.append(mu_t); sg_all.append(sg_t)
                    mu_e = mu_t.mean(0); s2 = (sg_t ** 2 + mu_t ** 2).mean(0) - mu_e ** 2
                    lbs['Deep_Ensemble'].append(np.clip(mu_e, 0, 125) - C.Z_SCORE * np.sqrt(np.clip(s2, 0, None)))
                # engine ids / unclipped labels, same ordering as extract_raw_windows(mode='test')
                units = [u for u in test_raw['unit_nr'].unique() if (test_raw['unit_nr'] == u).sum() >= C.SEQUENCE_LENGTH]
                y_unc = np.array([true_ruls.iloc[u - 1].item() - 1 for u in units], float)
                n_all = len(test_raw['unit_nr'].unique())
                rec['cond'][cond] = {'lb': {m: np.stack(v) for m, v in lbs.items()},
                                     'mu': np.stack(mu_all), 'sigma': np.stack(sg_all),   # (trials, seeds, engines)
                                     'y': np.array(ys[0]), 'y_unc': y_unc, 'units': np.array(units), 'n_test_engines_file': n_all}
                print('  ', cond, 'engines', len(units), 'of', n_all, flush=True)
            pickle.dump(rec, open(os.path.join(OUT, tag + '.pkl'), 'wb'))
            del mcm, nllm
            if dev.type == 'cuda': torch.cuda.empty_cache()
