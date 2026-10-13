"""JRR branch, exploratory inference-only control (NOT in the v1.8 pipeline).
Question: v1.8's drift ramp is one-directional (+k*range on every sensor channel) whereas the bias family and the
'equal-end bias' control use a random sign per channel.  The drift-vs-bias contrast therefore confounds temporal
structure with cross-channel sign coherence.  Here a CONSTANT offset of +k*range (or -k*range) is applied to every
sensor channel at k = 5% of range (no ramp), and the ramp is re-run as a check against the released numbers.
Heteroscedastic head only; five released checkpoints per backbone/sub-dataset; terminal window; L = 20.
Output: results_jrr/derived/coherent_bias_control.json
"""
import os, sys, json
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, 'code'))
import common as C, noise_injection as V4, transformer_common as T2, sweep_engine as E
from maintenance_decision_two_sided import get_raw_test_window
if __name__ == '__main__':
    C.require_fixed_hashseed(); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    canon = json.load(open(os.path.join(PROJ, 'results', 'canonical_splits.json'))); rng = np.random.RandomState(7)
    out = {}
    for bb in ['LSTM', 'Transformer']:
        for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
            tr_raw, _, _, fc, _ = V4.load_raw_train_test_and_scaler(ds)
            fs = V4.sensor_only_scale(fc, V4.fit_fullscale_range(tr_raw, fc)); mask = V4.sensor_mask_for(fc)
            X_clean, y, *_ = get_raw_test_window(ds, 'clean', 0, fs, None)
            k = fs * 0.05; T = X_clean.shape[1]
            variants = {'ramp': X_clean + np.linspace(0, 1, T)[None, :, None] * k[None, None, :], 'const_pos': X_clean + k[None, None, :], 'const_neg': X_clean - k[None, None, :], 'clean': X_clean}
            res = {}
            for vn, Xv in variants.items():
                lbs, mus, picps, foobs = [], [], [], []
                for s in C.SEEDS:
                    fu = canon[ds][str(s)]['fit_units']; _, _, _, _, sc = V4.load_raw_train_test_and_scaler_leakfree(ds, fu)
                    m = T2.load_checkpoint_model_t2(bb, T2.nll_ckpt_path(bb, ds, s), dev)
                    Xs = V4.scale_raw_windows(Xv, sc); mu, ls = E.infer_nll(m, torch.tensor(Xs, dtype=torch.float32).to(dev)); sg = np.exp(ls) * 125.0
                    lbs.append(mu - C.Z_SCORE * sg); mus.append(mu); picps.append(float(np.mean(np.abs(y - mu) <= C.Z_SCORE * sg))); foobs.append(V4.feat_oob(Xs, mask))
                lb = np.stack(lbs); mu = np.stack(mus); L = 20
                trig = lb <= L; prem = (trig & (y[None] > L + 20)).mean(0); unr = ((y[None] <= L) & ~trig).mean(0)
                idx = rng.randint(0, len(y), (2000, len(y)))
                ci = lambda v: [float(v.mean())] + [float(np.percentile(v[idx].mean(1), 2.5)), float(np.percentile(v[idx].mean(1), 97.5))]
                res[vn] = {'picp': float(np.mean(picps)), 'f_oob': float(np.mean(foobs)), 'mean_mu_shift_vs_clean': None, 'prem': ci(prem), 'unrec': ci(unr), 'mu_mean': float(mu.mean())}
            for vn in res: res[vn]['mean_mu_shift_vs_clean'] = res[vn]['mu_mean'] - res['clean']['mu_mean']
            out[f'{bb}|{ds}'] = res; print(bb, ds, {v: (round(r['picp'], 3), round(r['prem'][0], 3), round(r['unrec'][0], 3)) for v, r in res.items()}, flush=True)
    json.dump(out, open(os.path.join(PROJ, 'results_jrr', 'derived', 'coherent_bias_control.json'), 'w'), indent=1)
