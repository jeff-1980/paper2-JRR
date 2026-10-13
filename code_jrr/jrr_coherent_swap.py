"""JRR branch, exploratory inference-only extension of jrr_coherent_bias_control.py.
Same-condition mean/scale output swap (heteroscedastic head, L = 20, 2000 engine-bootstrap resamples) for the three
sign-pattern variants of the 5%-of-range offset: ramp (+), constant +, constant -.  Mean and scale are each taken from
the clean or the perturbed condition of the same model; rates are per-engine means over the five models.
Output: results_jrr/derived/coherent_swap.json
"""
import os, sys, json
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, 'code'))
import common as C, noise_injection as V4, transformer_common as T2, sweep_engine as E
from maintenance_decision_two_sided import get_raw_test_window
if __name__ == '__main__':
    C.require_fixed_hashseed(); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    canon = json.load(open(os.path.join(PROJ, 'results', 'canonical_splits.json'))); rng = np.random.RandomState(11); Z = C.Z_SCORE; L = 20
    out = {}
    for bb in ['LSTM', 'Transformer']:
        for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
            tr_raw, _, _, fc, _ = V4.load_raw_train_test_and_scaler(ds)
            fs = V4.sensor_only_scale(fc, V4.fit_fullscale_range(tr_raw, fc))
            X_clean, y, *_ = get_raw_test_window(ds, 'clean', 0, fs, None)
            k = fs * 0.05; T = X_clean.shape[1]
            variants = {'clean': X_clean, 'ramp': X_clean + np.linspace(0, 1, T)[None, :, None] * k[None, None, :],
                        'const_pos': X_clean + k[None, None, :], 'const_neg': X_clean - k[None, None, :]}
            MU, SG = {}, {}
            for vn, Xv in variants.items():
                mus, sgs = [], []
                for s in C.SEEDS:
                    _, _, _, _, sc = V4.load_raw_train_test_and_scaler_leakfree(ds, canon[ds][str(s)]['fit_units'])
                    m = T2.load_checkpoint_model_t2(bb, T2.nll_ckpt_path(bb, ds, s), dev)
                    mu, ls = E.infer_nll(m, torch.tensor(V4.scale_raw_windows(Xv, sc), dtype=torch.float32).to(dev)); mus.append(mu); sgs.append(np.exp(ls) * 125.0)
                MU[vn] = np.stack(mus); SG[vn] = np.stack(sgs)
            n = len(y); idx = rng.randint(0, n, (2000, n))
            def rates(mu, sg):
                tr = (mu - Z * sg) <= L
                return (tr & (y[None] > L + 20)).mean(0), ((y[None] <= L) & ~tr).mean(0)
            def stat(w): return [float(w.mean()), float(np.percentile(w[idx].mean(1), 2.5)), float(np.percentile(w[idx].mean(1), 97.5))]
            res = {}
            for vn in ['ramp', 'const_pos', 'const_neg']:
                R = {'00': rates(MU['clean'], SG['clean']), '10': rates(MU[vn], SG['clean']), '01': rates(MU['clean'], SG[vn]), '11': rates(MU[vn], SG[vn])}
                r = {'mean_shift': float((MU[vn] - MU['clean']).mean()), 'sigma_ratio': float((SG[vn] / SG['clean']).mean())}
                for kind, j in [('prem', 0), ('unrec', 1)]:
                    v = {q: R[q][j] for q in R}
                    r[kind] = {'clean': stat(v['00']), 'mu_pert': stat(v['10']), 'sigma_pert': stat(v['01']), 'both': stat(v['11']),
                               'delta_mu_given_sigma_clean': stat(v['10'] - v['00']), 'delta_sigma_given_mu_pert': stat(v['11'] - v['10'])}
                res[vn] = r
            out[f'{bb}|{ds}'] = res
            print(bb, ds, {vn: (round(res[vn]['mean_shift'], 1), round(res[vn]['prem']['both'][0], 3), round(res[vn]['unrec']['both'][0], 3)) for vn in res}, flush=True)
    json.dump(out, open(os.path.join(PROJ, 'results_jrr', 'derived', 'coherent_swap.json'), 'w'), indent=1)
