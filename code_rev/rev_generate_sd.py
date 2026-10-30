"""JRR revision: new inference-only analyses (no training).

For each backbone x sub-dataset, stores per-engine mean/scale outputs of the
heteroscedastic head (NLL), the fixed-variance head, MC Dropout and the deep
ensemble under the original and the new perturbation conditions, plus
calibration-set quantities for the calibration-unit control, and full-trajectory
outputs of the heteroscedastic head for the first-trigger analysis.

Run: PYTHONHASHSEED=0 RUL_DATA_DIR=<cmapss> python rev_generate.py <repo> <outdir> [BB/DS ...]
"""
import os, sys, json, pickle, time
import numpy as np, torch
REPO = os.path.abspath(sys.argv[1]); OUT = os.path.abspath(sys.argv[2])
sys.path.insert(0, os.path.join(REPO, 'code'))
import common as C
import noise_injection as V4
import transformer_common as T2
import sweep_engine as E
from maintenance_decision_two_sided import calib_sigma_fixed, sequences_for_units

os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(2)
DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']; BACKBONES = ['LSTM', 'Transformer']
SEQ = C.SEQUENCE_LENGTH
GROUPS = {  # C-MAPSS sensor groups (Saxena et al. 2008 sensor table)
    'temp': ['s_2', 's_3', 's_4'],              # T24, T30, T50
    'press': ['s_7', 's_11'],                   # P30, Ps30
    'speed': ['s_8', 's_9', 's_13', 's_14'],    # Nf, Nc, NRf, NRc
    'flowbleed': ['s_12', 's_15', 's_17', 's_20', 's_21'],  # phi, BPR, htBleed, W31, W32
}


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


def ramp_delta(seq_len, k_vec):
    return np.linspace(0.0, 1.0, seq_len)[None, :, None] * k_vec[None, None, :]


def continuous_windows(test_raw, feat, per_cycle_vec=None, end_vec=None):
    """Terminal windows after a drift on the whole test trajectory: either a
    0->end ramp over each trajectory (end_vec) or a constant rate per cycle
    from the first recorded cycle (per_cycle_vec)."""
    X = []
    for u in test_raw['unit_nr'].unique():
        d = test_raw[test_raw['unit_nr'] == u][feat].values.astype(np.float64)
        n = len(d)
        if n < SEQ: continue
        if end_vec is not None:
            delta = np.linspace(0.0, 1.0, n)[:, None] * end_vec[None, :]
        else:
            delta = np.arange(n)[:, None] * per_cycle_vec[None, :]
        X.append((d + delta)[-SEQ:])
    return np.array(X)


def within_sd(train_raw, feat):
    """Per-channel within-operating-condition standard deviation of the
    training file (mean over the KMeans conditions of the noise code)."""
    km, cs, _ = V4.fit_condition_model(train_raw, feat)
    return np.mean([cs[c] for c in cs], axis=0)


def build_conditions(ds, test_raw, feat, fs, sgn_deg, sdw=None):
    """Shifts expressed in units of the within-condition SD (sdw), sensor channels only."""
    Xc, y_ref, _ = V4.extract_raw_windows(test_raw, feat, TRUE_RULS[ds], mode='test')
    smask = fs > 0
    sd = np.where(smask, sdw, 0.0)
    sensor_idx = np.where(smask)[0]
    conds = {'clean': [Xc]}
    for k in (0.5, 1, 2, 4):
        conds[f'sd_drift{k}'] = [Xc + ramp_delta(SEQ, sd * k)]
        conds[f'sd_neg{k}'] = [Xc - ramp_delta(SEQ, sd * k)]
    for k in (1, 2):
        conds[f'sd_aligned{k}'] = [Xc + ramp_delta(SEQ, sd * k * sgn_deg)]
        conds[f'sd_anti{k}'] = [Xc - ramp_delta(SEQ, sd * k * sgn_deg)]
        tr = []
        for t in range(V4.N_TRIALS):
            rng = np.random.RandomState(C.stable_seed(ds, 'rev_sd_bias', k, t))
            sign = rng.choice([-1.0, 1.0], size=len(feat))
            tr.append(Xc + (sd * k * sign)[None, None, :])
        conds[f'sd_bias{k}'] = tr
    for kc in (1, 3):
        rng = np.random.RandomState(C.stable_seed(ds, 'rev_sd_kchan', kc))
        tr = []
        for _ in range(10):
            sel = rng.choice(sensor_idx, size=kc, replace=False)
            kv = np.zeros_like(sd); kv[sel] = sd[sel] * 2
            tr.append(Xc + ramp_delta(SEQ, kv))
        conds[f'sd_k{kc}drift2'] = tr
    # constant-rate trajectory drift reaching 2 SD after 100 cycles, all sensors positive / aligned
    conds['sd_rate2'] = [continuous_windows(test_raw, feat, per_cycle_vec=sd * 0.02)]
    conds['sd_rate2_aligned'] = [continuous_windows(test_raw, feat, per_cycle_vec=sd * 0.02 * sgn_deg)]
    return conds, np.array(y_ref)


TRUE_RULS = {}

if __name__ == '__main__':
    C.require_fixed_hashseed()
    dev = torch.device('cpu')
    canon = json.load(open(os.path.join(REPO, 'results', 'canonical_splits.json')))
    only = sys.argv[3:]
    for bb in BACKBONES:
        for ds in DATASETS:
            tag = f'{bb}_{ds}'
            if only and f'{bb}/{ds}' not in only: continue
            fout = os.path.join(OUT, tag + '.pkl')
            if os.path.exists(fout): continue
            t0 = time.time(); print('==', tag, flush=True)
            tr_raw, test_raw, true_ruls, feat, _ = V4.load_raw_train_test_and_scaler(ds)
            TRUE_RULS[ds] = true_ruls
            fs = V4.sensor_only_scale(feat, V4.fit_fullscale_range(tr_raw, feat))
            smask = V4.sensor_mask_for(feat)
            units = [u for u in test_raw['unit_nr'].unique() if (test_raw['unit_nr'] == u).sum() >= SEQ]
            y_unc = np.array([true_ruls.iloc[u - 1].item() - 1 for u in units], float)
            # full-trajectory windows of test engines (for first-trigger analysis)
            Xtraj_c, ytraj, utraj = V4.extract_raw_windows(test_raw, feat, true_ruls, mode='full_trajectory')
            # unclipped true RUL for every trajectory window
            ytraj_unc = []
            for u in units:
                n = (test_raw['unit_nr'] == u).sum(); r_end = true_ruls.iloc[u - 1].item() - 1
                ytraj_unc.extend([r_end + (n - 1 - (i + SEQ - 1)) for i in range(n - SEQ + 1)])
            ytraj_unc = np.array(ytraj_unc, float)
            # trajectory drift: constant 0.05%/cycle from first recorded cycle, all sensors positive
            seeds_rec = {}
            sgn = None
            rec = {'seeds': list(C.SEEDS), 'feat': feat, 'units': np.array(units), 'y_unc': y_unc,
                   'n_file': len(test_raw['unit_nr'].unique()), 'calib': {}, 'cond': {}, 'traj': {}}
            models = {}
            for s in C.SEEDS:
                fu = canon[ds][str(s)]['fit_units']; cu = canon[ds][str(s)]['calib_units']
                _, _, _, _, scal = V4.load_raw_train_test_and_scaler_leakfree(ds, fu)
                avar, mcm = calib_sigma_fixed(ds, bb, s, canon, dev)
                nllm = T2.load_checkpoint_model_t2(bb, T2.nll_ckpt_path(bb, ds, s), dev)
                models[s] = (scal, avar, mcm, nllm)
                if s == C.SEEDS[0]:
                    sgn = deg_direction(tr_raw, feat, fu) * smask
                    rec['deg_sign'] = sgn
                # calibration windows (all overlapping windows of the calibration engines)
                train_df, _, _, fcols, _ = C.load_and_process_leakfree(ds, fu)
                cal_df = train_df[train_df['unit_nr'].isin(cu)]
                Xcal, ycal = sequences_for_units(cal_df, fcols, cu)
                ucal = np.concatenate([[u] * (len(cal_df[cal_df['unit_nr'] == u]) - SEQ) for u in sorted(cu)])
                mu, ls = E.infer_nll(nllm, torch.tensor(Xcal, dtype=torch.float32))
                rec['calib'][s] = {'mu': mu, 'sigma': np.exp(ls) * 125.0, 'y': np.clip(ycal, 0, C.MAX_RUL),
                                   'unit': ucal}
            conds, y_ref = build_conditions(ds, test_raw, feat, fs, sgn, sdw=within_sd(tr_raw, feat))
            rec['sdw'] = within_sd(tr_raw, feat); rec['fs'] = fs
            rec['y'] = y_ref
            for cname, trials in conds.items():
                o = {k: [] for k in ('mu_nll', 'sg_nll', 'mu_mse', 'sg_mse', 'mu_mc', 'sg_mc', 'foob')}
                for t, Xr in enumerate(trials):
                    row = {k: [] for k in o}
                    for s in C.SEEDS:
                        scal, avar, mcm, nllm = models[s]
                        Xs_np = V4.scale_raw_windows(Xr, scal)
                        Xs = torch.tensor(Xs_np, dtype=torch.float32)
                        mu, ls = E.infer_nll(nllm, Xs)
                        row['mu_nll'].append(mu); row['sg_nll'].append(np.exp(ls) * 125.0)
                        m2, s2 = E.infer_mse_fixed(mcm, Xs, sigma_fixed=float(np.sqrt(avar)))
                        row['mu_mse'].append(m2); row['sg_mse'].append(s2)
                        m3, s3 = E.infer_mc_dropout(mcm, Xs, T=50, aleatory_var=avar,
                                                    seed=C.stable_seed(ds, bb, 'rev_' + cname, t, s, 'mc'))
                        row['mu_mc'].append(m3); row['sg_mc'].append(s3)
                        row['foob'].append(np.array([V4.feat_oob(Xs_np, smask)]))
                    for k in o: o[k].append(np.stack(row[k]))
                rec['cond'][cname] = {k: np.stack(v) for k, v in o.items()}  # (trials, seeds, engines)
                print('  ', cname, len(trials), f'{time.time()-t0:.0f}s', flush=True)
            pickle.dump(rec, open(fout, 'wb'))
