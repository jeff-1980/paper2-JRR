"""Writes rev_numbers.tex (macros) and rev_tables.tex (tables) for the revised manuscript from the
revision result files. No number in the new text is typed by hand.
Run: python make_rev_numbers.py <work dir> <out dir>
"""
import os, sys, json, pickle, csv
import numpy as np
from scipy.stats import beta

W, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
sd = json.load(open(os.path.join(W, 'res_sd', 'sd_conditions.json')))
sens = json.load(open(os.path.join(W, 'res_sd', 'sd_sensitivity_summary.json')))
rv = json.load(open(os.path.join(W, 'res', 'revision_results.json')))
pc = json.load(open(os.path.join(W, 'perchannel_sd.json')))
tj = json.load(open(os.path.join(W, 'traj_sd.json')))
G = {(r['backbone'], r['dataset'], r['condition']): r for r in sd}
DS = ['FD001', 'FD002', 'FD003', 'FD004']; BB = ['LSTM', 'Transformer']
M = {}


def f0(x): return f'{x:.0f}'
def f1(x): return f'{x:.1f}'
def f2(x): return f'{x:.2f}'


def rng(vals, fmt=f0, sep='--'):
    a, b = fmt(min(vals)), fmt(max(vals))
    return a if a == b else f'{a}{sep}{b}'


def vals(cond, key, ds=DS):
    return [r[key] for r in sd if r['condition'] == cond and r['dataset'] in ds]


# ---------- engine composition (shares of engines at risk / far from need), from the per-engine outputs
comp = {}
for bb in BB:
    for ds in DS:
        y = pickle.load(open(os.path.join(W, 'out_sd', f'{bb}_{ds}.pkl'), 'rb'))['y']
        comp[ds] = {'atrisk20': float(np.mean(y <= 20) * 100), 'far40': float(np.mean(y > 40) * 100), 'n': len(y)}
M['RVAtRisk'] = rng([c['atrisk20'] for c in comp.values()])
M['RVFar'] = rng([c['far40'] for c in comp.values()])
M['RVZeroUpperTwo'] = f1(beta.ppf(0.95, 1, comp['FD002']['n']) * 100)
M['RVZeroUpperFour'] = f1(beta.ppf(0.95, 1, comp['FD004']['n']) * 100)
M['RVZeroUpperOne'] = f1(beta.ppf(0.95, 1, comp['FD001']['n']) * 100)

# ---------- magnitude reference
ratio = {}
for ds in DS:
    r = pickle.load(open(os.path.join(W, 'out_sd', f'LSTM_{ds}.pkl'), 'rb'))
    m = r['fs'] > 0
    ratio[ds] = 0.05 * r['fs'][m] / r['sdw'][m]
M['RVRatioSingle'] = f2(np.median(np.concatenate([ratio['FD001'], ratio['FD003']])))
M['RVRatioSix'] = f0(np.median(np.concatenate([ratio['FD002'], ratio['FD004']])))
M['RVRatioFactor'] = f0(np.median(np.concatenate([ratio['FD002'], ratio['FD004']])) / np.median(np.concatenate([ratio['FD001'], ratio['FD003']])))
M['RVRatioSixRange'] = rng(np.concatenate([ratio['FD002'], ratio['FD004']]), f1)

# ---------- main SD conditions (heteroscedastic head, L = 20)
for tag, c in (('Half', 'sd_drift0.5'), ('One', 'sd_drift1'), ('Two', 'sd_drift2'), ('Four', 'sd_drift4')):
    M[f'RVPrem{tag}'] = rng(vals(c, 'NLL_L20_early'))
    M[f'RVPicp{tag}'] = rng(vals(c, 'picp90_NLL'), f2)
    M[f'RVCov{tag}'] = rng(vals(c, 'cov1_NLL'), f2)
    M[f'RVUnrecPos{tag}'] = rng(vals(c, 'NLL_L20_late'), f1)
for tag, c in (('One', 'sd_neg1'), ('Two', 'sd_neg2'), ('Four', 'sd_neg4')):
    M[f'RVLateNeg{tag}'] = rng(vals(c, 'NLL_L20_late'))
    M[f'RVPremNeg{tag}'] = rng(vals(c, 'NLL_L20_early'), f1)
    M[f'RVCovNeg{tag}'] = rng(vals(c, 'cov1_NLL'), f2)
    M[f'RVPicpNeg{tag}'] = rng(vals(c, 'picp90_NLL'), f2)
    cond_rate = [r['NLL_L20_late'] / comp[r['dataset']]['atrisk20'] * 100 for r in sd if r['condition'] == c]
    M[f'RVLateNegCond{tag}'] = rng(cond_rate)
M['RVFoobTwo'] = f1(max(vals('sd_drift2', 'foob') + vals('sd_neg2', 'foob')) * 100)
M['RVFoobAlignedTwo'] = f1(max(vals('sd_aligned2', 'foob') + vals('sd_anti2', 'foob')) * 100)
M['RVPremAligned'] = rng(vals('sd_aligned2', 'NLL_L20_early'))
M['RVLateAnti'] = rng(vals('sd_anti2', 'NLL_L20_late'))
M['RVPremBiasTwo'] = rng(vals('sd_bias2', 'NLL_L20_early'))
M['RVLateBiasTwo'] = rng(vals('sd_bias2', 'NLL_L20_late'), f1)
M['RVPicpBiasTwo'] = rng(vals('sd_bias2', 'picp90_NLL'), f2)
M['RVCovBiasTwo'] = rng(vals('sd_bias2', 'cov1_NLL'), f2)
M['RVPremRate'] = rng(vals('sd_rate2', 'NLL_L20_early'))
M['RVPremRateAligned'] = rng(vals('sd_rate2_aligned', 'NLL_L20_early'))
M['RVCleanPrem'] = rng(vals('clean', 'NLL_L20_early'))
M['RVCleanLate'] = rng(vals('clean', 'NLL_L20_late'), f1)
M['RVCleanCovOne'] = rng(vals('clean', 'cov1_NLL'), f2)
M['RVPremTwoSingle'] = rng(vals('sd_drift2', 'NLL_L20_early', ['FD001', 'FD003']))
M['RVPremTwoSix'] = rng(vals('sd_drift2', 'NLL_L20_early', ['FD002', 'FD004']))
M['RVPremOneSingle'] = rng(vals('sd_drift1', 'NLL_L20_early', ['FD001', 'FD003']))
M['RVPremOneSix'] = rng(vals('sd_drift1', 'NLL_L20_early', ['FD002', 'FD004']))
allm = ['NLL', 'MSE_fixed', 'MC_Dropout_fixed', 'Deep_Ensemble', 'CP_norm']
M['RVPremTwoAllMech'] = rng([r[f'{m}_L20_early'] for r in sd if r['condition'] == 'sd_drift2' for m in allm])
M['RVLateTwoAllMech'] = rng([r[f'{m}_L20_late'] for r in sd if r['condition'] == 'sd_neg2' for m in allm])
M['RVUnrecTwoAllMechMax'] = f1(max(r[f'{m}_L20_late'] for r in sd if r['condition'] in ('sd_drift2', 'sd_drift1') for m in allm))
M['RVPremNegAllMechMax'] = f1(max(r[f'{m}_L20_early'] for r in sd if r['condition'] in ('sd_neg2', 'sd_neg1') for m in allm))
# engine-bootstrap CI half-width at +2 SD
hw = [(r['NLL_L20_early_hi'] - r['NLL_L20_early_lo']) / 2 for r in sd if r['condition'] == 'sd_drift2']
M['RVPremTwoHalfCI'] = f0(max(hw))

# channel subsets
M['RVKOneMed'] = rng(vals('sd_k1drift2', 'NLL_L20_early_subset_median'), f0)
M['RVKOneMax'] = f0(max(vals('sd_k1drift2', 'NLL_L20_early_subset_max')))
M['RVKThreeMed'] = rng(vals('sd_k3drift2', 'NLL_L20_early_subset_median'), f0)
M['RVKThreeMax'] = f0(max(vals('sd_k3drift2', 'NLL_L20_early_subset_max')))
M['RVKThreeLateMax'] = f1(max(vals('sd_k3drift2', 'NLL_L20_late_subset_max')))
pos = [x for x in pc if x['sign'] == 1]
M['RVChanN'] = str(len(pos))
M['RVChanGtTen'] = str(sum(x['premature_L20'] > 10 for x in pos))
M['RVChanMax'] = f0(max(x['premature_L20'] for x in pos))
M['RVChanMaxSingleCond'] = f0(max(x['premature_L20'] for x in pos if x['dataset'] in ('FD001', 'FD003')))
M['RVChanMaxSixCond'] = f0(max(x['premature_L20'] for x in pos if x['dataset'] in ('FD002', 'FD004')))
mat = 0; flip = 0
for x in pos:
    m_ = [z for z in pc if z['backbone'] == x['backbone'] and z['dataset'] == x['dataset'] and z['channel'] == x['channel'] and z['sign'] == -1][0]
    if x['premature_L20'] > 10 or x['unrecognised_L20'] > 3:
        mat += 1
        sp = x['premature_L20'] > x['unrecognised_L20']; sm = m_['premature_L20'] > m_['unrecognised_L20']
        flip += sp != sm
M['RVChanMaterial'] = str(mat); M['RVChanFlip'] = str(flip)
neg_prem = [x for x in pc if x['sign'] == -1 and x['premature_L20'] > 10]
M['RVChanNegPremN'] = str(len(neg_prem))
M['RVChanNegPremMax'] = f0(max([x['premature_L20'] for x in neg_prem] or [0]))

# ---------- trajectory first trigger
T = {(r['backbone'], r['dataset'], r['scenario'], r['L']): r for r in tj}
def tv(scen, key, L=20, ds=DS): return [r[key] for r in tj if r['scenario'] == scen and r['L'] == L and r['dataset'] in ds]
M['RVTrajCleanPrem'] = rng(tv('clean', 'premature_first_pct'))
M['RVTrajPosPrem'] = rng(tv('rate_pos', 'premature_first_pct'))
M['RVTrajAlignedPrem'] = rng(tv('rate_aligned', 'premature_first_pct'))
M['RVTrajAntiPrem'] = rng(tv('rate_anti', 'premature_first_pct'), f1)
M['RVTrajCleanMiss'] = rng(tv('clean', 'missed_among_reaching_L_pct'))
M['RVTrajAntiMiss'] = rng(tv('rate_anti', 'missed_among_reaching_L_pct'))
M['RVTrajPosMiss'] = rng(tv('rate_pos', 'missed_among_reaching_L_pct'))
fg = [x for x in tv('rate_aligned', 'mean_forgone_cycles') if x is not None]
M['RVTrajAlignedForgone'] = rng(fg)
fgc = [x for x in tv('clean', 'mean_forgone_cycles') if x is not None]
M['RVTrajCleanForgone'] = rng(fgc)
M['RVTrajReach'] = rng(tv('clean', 'engines_reaching_L'))

# ---------- calibration unit (clean), range-scale results file
cu = rv['calibration_unit']
M['RVCovOneNLL'] = rng([r['cov1_NLL'] * 100 for r in cu], f1)
M['RVCovOneCP'] = rng([r['cov1_CP_norm'] * 100 for r in cu], f1)
M['RVCovOneEns'] = rng([r['cov1_Deep_Ensemble'] * 100 for r in cu], f1)
M['RVCovOneFix'] = rng([r['cov1_MSE_fixed'] * 100 for r in cu], f1)
M['RVCovOneMC'] = rng([r['cov1_MC_Dropout_fixed'] * 100 for r in cu], f1)
M['RVCovOneNLLBelow'] = str(sum(r['cov1_NLL_hi'] < 0.95 for r in cu))
M['RVCovOneCPBelow'] = str(sum(r['cov1_CP_norm_hi'] < 0.95 for r in cu))
M['RVCovOneEW'] = rng([r['cov1_CP_engine_weighted'] * 100 for r in cu], f1)
M['RVCovOneLM'] = rng([r['cov1_CP_label_matched'] * 100 for r in cu], f1)
M['RVCovOneOPE'] = rng([r['cov1_CP_one_per_engine_mean'] * 100 for r in cu], f1)
M['RVQAll'] = rng([r['q_CP_all_windows'] for r in cu], f2)
M['RVQEW'] = rng([r['q_CP_engine_weighted'] for r in cu], f2)
M['RVQOPE'] = rng([r['q_CP_one_per_engine_mean'] for r in cu], f2)
M['RVCalEng'] = rng([r['n_calib_engines'] for r in cu])
M['RVCalWin'] = rng([r['n_calib_windows'] for r in cu])
d = [abs(r[f'CP_{v}_drift5_early'] - r['CP_all_windows_drift5_early']) for r in cu for v in ('engine_weighted', 'label_matched')
     if r['dataset'] in ('FD002', 'FD004')]
M['RVCalDriftShiftSix'] = f1(max(d))
d1 = [r[f'CP_{v}_{c}_late'] - r[f'CP_all_windows_{c}_late'] for r in cu for v in ('engine_weighted', 'label_matched') for c in ('negdrift5', 'bias5')]
M['RVCalLateShift'] = rng(d1, f1, ' to ')
d2 = [r[f'CP_{v}_clean_early'] - r['CP_all_windows_clean_early'] for r in cu for v in ('engine_weighted', 'label_matched')]
M['RVCalCleanPremShift'] = rng(d2, f1, ' to ')
M['RVCPqDiff'] = f'{rv["cp_q_max_abs_diff_vs_release"]:.4f}'

# ---------- sensitivity (SD conditions)
s05 = [s for s in sens if s['alpha'] == 0.05 and s['tol'] == 20][0]
M['RVSensDir'] = s05['pos_premature_dominant'].split('/')[1]
M['RVSensAllOK'] = 'all' if all(s['pos_premature_dominant'].split('/')[0] == s['pos_premature_dominant'].split('/')[1] and
                                s['neg_late_dominant'].split('/')[0] == s['neg_late_dominant'].split('/')[1] for s in sens) else 'not all'
M['RVSensPremRange'] = rng([v for s in sens for v in s['NLL_L20_premature_sd_drift2_range']])
M['RVSensPremTolForty'] = rng([v for s in sens if s['tol'] == 40 for v in s['NLL_L20_premature_sd_drift2_range']])
M['RVSensPremAlphaOne'] = rng([v for s in sens if s['alpha'] == 0.01 for v in s['NLL_L20_premature_sd_drift2_range']])
M['RVSensLateRange'] = rng([v for s in sens for v in s['NLL_L20_late_sd_neg2_range']])
M['RVRankGroups'] = str(s05['rank_groups']); M['RVRankChg'] = str(s05['ranking_changes']); M['RVTopChg'] = str(s05['top_changes'])
M['RVRegretTen'] = str(s05['relative_regret_gt10pct'])
M['RVRankChgRange'] = rng([s['ranking_changes'] for s in sens]); M['RVTopChgRange'] = rng([s['top_changes'] for s in sens])
M['RVRegretTenRange'] = rng([s['relative_regret_gt10pct'] for s in sens])

# ---------- output swap at +/-2 SD (mean vs scale), heteroscedastic head, L = 20
sw_mu = []; sw_sg = []; sw_mu_n = []; sw_sg_n = []
for bb in BB:
    for ds in DS:
        r = pickle.load(open(os.path.join(W, 'out_sd', f'{bb}_{ds}.pkl'), 'rb')); y = r['y']
        c0 = r['cond']['clean']
        for cond, lst_mu, lst_sg, early in (('sd_drift2', sw_mu, sw_sg, True), ('sd_neg2', sw_mu_n, sw_sg_n, False)):
            c1 = r['cond'][cond]
            def rate(mu, sg):
                lb = mu - 1.645 * sg; trig = lb <= 20
                return float(((trig & (y[None] > 40)) if early else ((~trig) & (y[None] <= 20))).mean() * 100)
            base = rate(c0['mu_nll'][0], c0['sg_nll'][0]); mu_only = rate(c1['mu_nll'][0], c0['sg_nll'][0])
            full = rate(c1['mu_nll'][0], c1['sg_nll'][0])
            lst_mu.append(mu_only - base); lst_sg.append(full - mu_only)
M['RVSwapMu'] = rng(sw_mu); M['RVSwapSg'] = rng(sw_sg, f1, ' to ')
M['RVSwapMuNeg'] = rng(sw_mu_n); M['RVSwapSgNeg'] = rng(sw_sg_n, f1, ' to ')

with open(os.path.join(OUT, 'rev_numbers.tex'), 'w') as fh:
    fh.write('% generated by make_rev_numbers.py -- do not edit by hand\n')
    for k, v in M.items():
        fh.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')

# ---------- Table: SD-unit stress test, heteroscedastic head, L = 20
L = []
L.append('\\begin{tabular}{llccccccc}\\toprule')
L.append(' & & \\multicolumn{3}{c}{Coverage} & \\multicolumn{2}{c}{Premature (\\%)} & \\multicolumn{2}{c}{Unrecognised (\\%)} \\\\')
L.append('\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}')
L.append('Cell & Shift & PICP & One-sided & $f_{\\mathrm{oob}}$ & rate & [95\\% CI] & rate & [95\\% CI] \\\\ \\midrule')
for bb in BB:
    for ds in DS:
        for c, lab in (('clean', 'none'), ('sd_drift2', '$+2$ SD ramp'), ('sd_neg2', '$-2$ SD ramp'), ('sd_bias2', '$\\pm2$ SD offset')):
            r = G[(bb, ds, c)]
            cell = f'{bb} {ds}' if c == 'clean' else ''
            L.append(f"{cell} & {lab} & {r['picp90_NLL']:.2f} & {r['cov1_NLL']:.2f} & {r['foob']*100:.1f} & {r['NLL_L20_early']:.0f} & "
                     f"[{r['NLL_L20_early_lo']:.0f}, {r['NLL_L20_early_hi']:.0f}] & {r['NLL_L20_late']:.1f} & "
                     f"[{r['NLL_L20_late_lo']:.1f}, {r['NLL_L20_late_hi']:.1f}] \\\\")
        if not (bb == 'Transformer' and ds == 'FD004'): L.append('\\addlinespace[2pt]')
L.append('\\bottomrule\\end{tabular}')
open(os.path.join(OUT, 'rev_table_sd.tex'), 'w').write('\n'.join(L) + '\n')

# ---------- Supplement table: calibration unit
L = ['\\begin{tabular}{llcccccccc}\\toprule',
     ' & & \\multicolumn{5}{c}{One-sided coverage of the 95\\% lower bound, clean} & \\multicolumn{3}{c}{CP-norm, other calibration units} \\\\',
     '\\cmidrule(lr){3-7}\\cmidrule(lr){8-10}',
     'Backbone & Data & Fixed var. & Heterosc. & MC Drop. & Ensemble & CP-norm & Engine-wt. & Label-matched & One/engine \\\\ \\midrule']
for r in cu:
    L.append(f"{r['backbone']} & {r['dataset']} & {r['cov1_MSE_fixed']:.3f} & {r['cov1_NLL']:.3f} [{r['cov1_NLL_lo']:.2f}, {r['cov1_NLL_hi']:.2f}] & "
             f"{r['cov1_MC_Dropout_fixed']:.3f} & {r['cov1_Deep_Ensemble']:.3f} & {r['cov1_CP_norm']:.3f} [{r['cov1_CP_norm_lo']:.2f}, {r['cov1_CP_norm_hi']:.2f}] & "
             f"{r['cov1_CP_engine_weighted']:.3f} & {r['cov1_CP_label_matched']:.3f} & {r['cov1_CP_one_per_engine_mean']:.3f} \\\\")
L.append('\\bottomrule\\end{tabular}')
open(os.path.join(OUT, 'rev_table_calib.tex'), 'w').write('\n'.join(L) + '\n')

# ---------- Supplement table: trajectory first trigger, L = 20
L = ['\\begin{tabular}{llcccccc}\\toprule',
     'Cell & Scenario & Premature first (\\%) & Timely first (\\%) & Never (\\%) & Reaching $L$ & Missed among reaching (\\%) & Forgone (cycles) \\\\ \\midrule']
for bb in BB:
    for ds in DS:
        for sc, lab in (('clean', 'clean'), ('rate_pos', '$+$ all'), ('rate_aligned', 'aligned'), ('rate_anti', 'anti-aligned')):
            r = T[(bb, ds, sc, 20)]
            fgv = '--' if r['mean_forgone_cycles'] is None else f"{r['mean_forgone_cycles']:.0f}"
            L.append(f"{bb + ' ' + ds if sc == 'clean' else ''} & {lab} & {r['premature_first_pct']:.1f} & {r['timely_first_pct']:.1f} & "
                     f"{r['never_pct']:.1f} & {r['engines_reaching_L']:.0f} & {r['missed_among_reaching_L_pct']:.0f} & {fgv} \\\\")
        L.append('\\addlinespace[2pt]')
L.append('\\bottomrule\\end{tabular}')
open(os.path.join(OUT, 'rev_table_traj.tex'), 'w').write('\n'.join(L) + '\n')

# ---------- Supplement table: sensitivity summary
L = ['\\begin{tabular}{ccccccc}\\toprule',
     '$\\alpha$ & Band & $+2$ SD: premature $>$ unrec. & $-2$ SD: unrec. $>$ premature & Premature, $+2$ SD (\\%) & Ranking changes & Top changes / regret $>10\\%$ \\\\ \\midrule']
for s in sens:
    L.append(f"{s['alpha']:.2f} & {s['tol']} & {s['pos_premature_dominant']} & {s['neg_late_dominant']} & "
             f"{s['NLL_L20_premature_sd_drift2_range'][0]:.0f}--{s['NLL_L20_premature_sd_drift2_range'][1]:.0f} & "
             f"{s['ranking_changes']}/{s['rank_groups']} & {s['top_changes']} / {s['relative_regret_gt10pct']} \\\\")
L.append('\\bottomrule\\end{tabular}')
open(os.path.join(OUT, 'rev_table_sens.tex'), 'w').write('\n'.join(L) + '\n')
for k, v in M.items(): print(k, v)
