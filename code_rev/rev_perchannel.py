"""JRR revision: single-channel ramps (+/-5% of range, or +/-2 within-condition SD with mode sd), every sensor channel, heteroscedastic head only.
Run: PYTHONHASHSEED=0 RUL_DATA_DIR=<cmapss> python rev_perchannel.py <repo> <out.json>
"""
import os, sys, json
import numpy as np, torch
REPO = os.path.abspath(sys.argv[1]); OUT = sys.argv[2]; MODE = sys.argv[3] if len(sys.argv) > 3 else 'range'
sys.path.insert(0, os.path.join(REPO, 'code'))
import common as C
import noise_injection as V4
import transformer_common as T2
import sweep_engine as E
torch.set_num_threads(1)
SEQ = C.SEQUENCE_LENGTH
res = []
if __name__ == '__main__':
    C.require_fixed_hashseed()
    canon = json.load(open(os.path.join(REPO, 'results', 'canonical_splits.json')))
    for bb in ['LSTM', 'Transformer']:
        for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
            tr_raw, test_raw, true_ruls, feat, _ = V4.load_raw_train_test_and_scaler(ds)
            fs = V4.sensor_only_scale(feat, V4.fit_fullscale_range(tr_raw, feat))
            km, cs, _ = V4.fit_condition_model(tr_raw, feat)
            sdw = np.where(fs > 0, np.mean([cs[c] for c in cs], 0), 0.0)
            amp = sdw * 2.0 if MODE == 'sd' else fs * 0.05
            Xc, y, _ = V4.extract_raw_windows(test_raw, feat, true_ruls, mode='test'); y = np.array(y)
            ramp = np.linspace(0, 1, SEQ)[None, :]
            ms = []
            for s in C.SEEDS:
                fu = canon[ds][str(s)]['fit_units']
                _, _, _, _, scal = V4.load_raw_train_test_and_scaler_leakfree(ds, fu)
                ms.append((scal, T2.load_checkpoint_model_t2(bb, T2.nll_ckpt_path(bb, ds, s), 'cpu')))
            def run(X):
                mu, sg = [], []
                for scal, m in ms:
                    a, l = E.infer_nll(m, torch.tensor(V4.scale_raw_windows(X, scal), dtype=torch.float32))
                    mu.append(a); sg.append(np.exp(l) * 125)
                return np.array(mu), np.array(sg)
            mu0, _ = run(Xc)
            for j, c in enumerate(feat):
                if fs[j] == 0: continue
                for sign in (+1, -1):
                    X = Xc.copy(); X[:, :, j] += sign * ramp * amp[j]
                    mu, sg = run(X); lb = mu - 1.645 * sg
                    res.append({'backbone': bb, 'dataset': ds, 'channel': c, 'sign': sign,
                                'dmu': float((mu - mu0).mean()),
                                'premature_L20': float(np.mean((lb <= 20) & (y[None] > 40)) * 100),
                                'unrecognised_L20': float(np.mean((lb > 20) & (y[None] <= 20)) * 100)})
            print(bb, ds, flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)
