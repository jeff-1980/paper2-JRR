"""JRR revision: first trigger along each test trajectory (heteroscedastic head, one-sided 95% bound),
clean and under a constant-rate drift of 0.02 within-condition SD per cycle from the first recorded
cycle (2 SD after 100 cycles): all sensors positive, degradation-aligned, anti-aligned.
Run: PYTHONHASHSEED=0 RUL_DATA_DIR=<cmapss> python rev_traj_sd.py <repo> <out.json>
"""
import os, sys, json
import numpy as np, torch
REPO = os.path.abspath(sys.argv[1]); OUT = sys.argv[2]
sys.path.insert(0, os.path.join(REPO, 'code')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, noise_injection as V4, transformer_common as T2, sweep_engine as E
torch.set_num_threads(2)
SEQ = C.SEQUENCE_LENGTH
rows = []


def deg_direction(train_raw, feat, fit_units):
    """+1 if the channel rises towards failure, -1 if it falls: sign of the
    Spearman correlation with -RUL on the fitting engines, after removing the
    operating-condition mean (KMeans on the settings, as in the noise code)."""
    from scipy.stats import spearmanr
    df = train_raw[train_raw['unit_nr'].isin(fit_units)].copy()
    rul = df.groupby('unit_nr')['time_cycles'].transform('max') - df['time_cycles']
    vals = df[feat].values.astype(float)
    km, _, _ = V4.fit_condition_model(train_raw, feat)
    lab = km.predict(df[C.SETTING_NAMES].values)
    for c in np.unique(lab):
        m = lab == c; vals[m] -= vals[m].mean(0)
    sgn = np.array([np.sign(spearmanr(vals[:, j], -rul.values)[0]) for j in range(len(feat))])
    sgn[np.isnan(sgn)] = 0
    return sgn



if __name__ == '__main__':
    C.require_fixed_hashseed()
    canon = json.load(open(os.path.join(REPO, 'results', 'canonical_splits.json')))
    for bb in ['LSTM', 'Transformer']:
        for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
            tr, te, tru, feat, _ = V4.load_raw_train_test_and_scaler(ds)
            fs = V4.sensor_only_scale(feat, V4.fit_fullscale_range(tr, feat)); smask = fs > 0
            km, cs, _ = V4.fit_condition_model(tr, feat); sdw = np.where(smask, np.mean([cs[c] for c in cs], 0), 0)
            sgn = deg_direction(tr, feat, canon[ds][str(C.SEEDS[0])]['fit_units']) * smask
            units = [u for u in te['unit_nr'].unique() if (te['unit_nr'] == u).sum() >= SEQ]
            scen = {'clean': np.zeros_like(sdw), 'rate_pos': sdw * 0.02, 'rate_aligned': sdw * 0.02 * sgn,
                    'rate_anti': -sdw * 0.02 * sgn}
            Xs = {k: [] for k in scen}; U = []; Y = []
            for u in units:
                d = te[te['unit_nr'] == u][feat].values.astype(float); n = len(d); r_end = tru.iloc[u - 1].item() - 1
                for k, v in scen.items():
                    dd = d + np.arange(n)[:, None] * v[None, :]
                    Xs[k].extend(dd[i:i + SEQ] for i in range(n - SEQ + 1))
                U.extend([u] * (n - SEQ + 1)); Y.extend(r_end + (n - SEQ - i) for i in range(n - SEQ + 1))
            U = np.array(U); Y = np.array(Y, float)
            models = []
            for s in C.SEEDS:
                _, _, _, _, scal = V4.load_raw_train_test_and_scaler_leakfree(ds, canon[ds][str(s)]['fit_units'])
                models.append((scal, T2.load_checkpoint_model_t2(bb, T2.nll_ckpt_path(bb, ds, s), 'cpu')))
            for k in scen:
                X = np.array(Xs[k]); lbs = []
                for scal, m in models:
                    mu, ls = E.infer_nll(m, torch.tensor(V4.scale_raw_windows(X, scal), dtype=torch.float32))
                    lbs.append(mu - 1.645 * np.exp(ls) * 125)
                lbs = np.array(lbs)
                for L in (10, 20, 30):
                    e = t = nv = ms = ar = 0; fg = []
                    for si in range(5):
                        for u in units:
                            ix = np.where(U == u)[0]; trig = np.where(lbs[si, ix] <= L)[0]; reach = Y[ix[-1]] <= L
                            ar += reach
                            if len(trig) == 0: nv += 1; ms += reach
                            elif Y[ix[trig[0]]] > L + 20: e += 1; fg.append(Y[ix[trig[0]]] - L)
                            else: t += 1
                    n = 5 * len(units)
                    rows.append({'backbone': bb, 'dataset': ds, 'scenario': k, 'L': L, 'n_engines': len(units),
                                 'premature_first_pct': e / n * 100, 'timely_first_pct': t / n * 100,
                                 'never_pct': nv / n * 100, 'missed_among_reaching_L_pct': ms / max(ar, 1) * 100,
                                 'engines_reaching_L': ar / 5, 'mean_forgone_cycles': float(np.mean(fg)) if fg else None})
            print(bb, ds, flush=True)
    json.dump(rows, open(OUT, 'w'), indent=1)
