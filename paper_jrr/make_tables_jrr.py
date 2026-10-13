"""JRR branch: tables and text macros that are NEW or RE-DERIVED relative to v1.8.
Run order (from repo root):
  python paper_jrr/make_tables_v18.py .        # unchanged v1.8 generator -> paper_jrr/tables/ (N-macros, v1.8 tables)
  python code_jrr/jrr_decision_per_engine.py   # inference-only per-engine re-derivation (needs C-MAPSS + checkpoints)
  python code_jrr/jrr_derived_stats.py         # engine bootstrap, output swap, label-instant, f_oob points
  python paper_jrr/make_tables_jrr.py .        # this file -> paper_jrr/tables/jrr_*.tex and numbers_jrr.tex (J-macros)
No number is typed by hand; every J-macro below is computed from results/ or results_jrr/derived/.
"""
import json, os, sys, csv, collections
import numpy as np
ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'
R = os.path.join(ROOT, 'results'); D = os.path.join(ROOT, 'results_jrr', 'derived')
T = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables')
def J(p, base=R): return json.load(open(os.path.join(base, p)))
BB = ['LSTM', 'Transformer']; DS = ['FD001', 'FD002', 'FD003', 'FD004']; SIX = [(b, d) for b in BB for d in ['FD002', 'FD004']]
METH = [('MSE_fixed', 'Fixed variance'), ('NLL', 'Heteroscedastic'), ('MC_Dropout_fixed', 'MC Dropout'), ('Deep_Ensemble', 'Deep ensemble'), ('CP_norm', 'Split-CP (norm)')]
M = {}
def mac(k, v): M[k] = v
def W(name, body): open(os.path.join(T, name + '.tex'), 'w').write(body)
def pc(x, d=0): return f"{100*x:.{d}f}"
def rng(vals, d=0, lo=None):
    a, b = f"{min(vals):.{d}f}", f"{max(vals):.{d}f}"
    return a if a == b else f"{a}--{b}"
boot = J('decision_engine_bootstrap.json', D); swap = J('output_swap.json', D); rk = J('rank_stability.json', D)
lab = J('label_instant_sensitivity.json', D); agree = J('agreement_with_release.json', D)
pe = {}
import pickle
for b in BB:
    for d in DS:
        pe[(b, d)] = pickle.load(open(os.path.join(D, 'per_engine', f'{b}_{d}.pkl'), 'rb'))
def B(bb, ds, cond, m='NLL', L=20): return boot[f'{bb}|{ds}|{cond}|{m}|L{L}']
# ---------- engines -------------------------------------------------------------------------------
ne = {d: len(pe[('LSTM', d)]['cond']['clean']['y']) for d in DS}; nf = {d: int(pe[('LSTM', d)]['cond']['clean']['n_test_engines_file']) for d in DS}
mac('EvalEng', f"{ne['FD001']}, {ne['FD002']}, {ne['FD003']} and {ne['FD004']}"); mac('FileEng', f"{nf['FD001']}, {nf['FD002']}, {nf['FD003']} and {nf['FD004']}")
mac('ExclTwo', str(nf['FD002'] - ne['FD002'])); mac('ExclFour', str(nf['FD004'] - ne['FD004']))
mac('AgreeN', str(agree['n_values_compared'])); mac('AgreeMax', f"{agree['max_abs_difference']:.0e}")
# ---------- drift / bias decision results (heteroscedastic head, L=20, six-condition) -----------------------
pr = [B(b, d, 'drift5pct')['prem'] for b, d in SIX]; mac('DriftPremRange', rng([100*x[0] for x in pr]))
mac('DriftPremCI', f"{100*min(x[1] for x in pr):.0f}--{100*max(x[2] for x in pr):.0f}")
mac('DriftPremModelRange', f"{100*min(min(B(b,d,'drift5pct')['prem_by_model']) for b,d in SIX):.0f}--{100*max(max(B(b,d,'drift5pct')['prem_by_model']) for b,d in SIX):.0f}")
mac('DriftPremHalfCI', f"{100*max((x[2]-x[1])/2 for x in pr):.0f}")
for L, Lw in ((10, 'ten'), (30, 'thirty')):
    mac(f'DriftPremL{Lw}', rng([100*B(b, d, 'drift5pct', 'NLL', L)['prem'][0] for b, d in SIX]))
allm = [B(b, d, 'drift5pct', m)['prem'][0] for b, d in SIX for m, _ in METH]; mac('DriftPremAllMech', rng([100*x for x in allm]))
mac('DriftUnrecMax', f"{100*max(B(b,d,'drift5pct',m,L)['unrec'][0] for b in BB for d in DS for m,_ in METH for L in (10,20,30)):.1f}")
bp = [B(b, d, 'bias5pct')['prem'][0] for b, d in SIX]; bu = [B(b, d, 'bias5pct')['unrec'][0] for b, d in SIX]; buc = [B(b, d, 'bias5pct')['unrec_cond'][0] for b, d in SIX]
mac('BiasPremRange', rng([100*x for x in bp])); mac('BiasUnrecRange', rng([100*x for x in bu], 1)); mac('BiasUnrecCondRange', rng([100*x for x in buc]))
mac('BiasUnrecMaxCI', f"{100*max(B(b,d,'bias5pct')['unrec'][2] for b,d in SIX):.1f}")
mac('BiasPremAllMech', rng([100*B(b, d, 'bias5pct', m)['prem'][0] for b, d in SIX for m, _ in METH]))
mac('BiasUnrecAllMech', rng([100*B(b, d, 'bias5pct', m)['unrec'][0] for b, d in SIX for m, _ in METH], 1))
ru = [B(b, d, 'drift5pct')['rul_unc_at_trigger'] for b, d in SIX]; mac('DriftRULunc', rng([x[0] for x in ru])); 
cap = []
for b, d in SIX:
    c = pe[(b, d)]['cond']['drift5pct']; lb = c['lb']['NLL']; tr = lb <= 20; pm = tr & (c['y'][None, :] > 40)
    cap.append(float(np.broadcast_to(c['y'][None, :], lb.shape)[pm].mean()))
mac('DriftRULcap', rng(cap))
mac('CleanPremRange', rng([100*B(b, d, 'clean')['prem'][0] for b, d in SIX])); mac('NoisePremRange', rng([100*B(b, d, 'gaussian1pct')['prem'][0] for b, d in SIX]))
mac('CleanUnrecMax', f"{100*max(B(b,d,'clean')['unrec'][0] for b,d in SIX):.1f}")
sp = [B(b, d, 'drift5pct')['prem'][0] for b in BB for d in ['FD001', 'FD003']]
mac('DriftPremSingle', f"{100*B('LSTM','FD001','drift5pct')['prem'][0]:.1f}, {100*B('Transformer','FD001','drift5pct')['prem'][0]:.1f}")
mac('DriftPremSingleThree', f"{100*B('LSTM','FD003','drift5pct')['prem'][0]:.0f} and {100*B('Transformer','FD003','drift5pct')['prem'][0]:.0f}")
# ---------- output swap ----------------------------------------------------------------------------------------
def S(b, d, c, L=20): return swap[f'{b}|{d}|{c}|L{L}']
dm = [S(b, d, 'drift5pct')['prem']['delta_mu_given_sigma_clean'][0] for b, d in SIX]; mac('SwapMuDrift', rng([100*x for x in dm]))
dmlo = [S(b, d, 'drift5pct')['prem']['delta_mu_given_sigma_clean'][1][0] for b, d in SIX]; mac('SwapMuDriftLoCI', f"{100*min(dmlo):.0f}")
ds_ = [S(b, d, 'drift5pct')['prem']['delta_sigma_given_mu_pert'][0] for b, d in SIX]; mac('SwapSigDrift', f"{100*min(ds_):+.0f}$ to ${100*max(ds_):+.0f}")
mac('MuShiftDrift', f"{min(S(b,d,'drift5pct')['mean_mu_shift'] for b,d in SIX):.0f}$ to ${max(S(b,d,'drift5pct')['mean_mu_shift'] for b,d in SIX):.0f}")
dm1 = [S(b, d, 'drift5pct')['prem']['delta_mu_given_sigma_clean'][0] for b in BB for d in ['FD001', 'FD003']]
ds1 = [S(b, d, 'drift5pct')['prem']['delta_sigma_given_mu_pert'][0] for b in BB for d in ['FD001', 'FD003']]
mac('SwapMuDriftSingle', rng([100*x for x in dm1])); mac('SwapSigDriftSingle', f"{100*min(ds1):+.0f}$ to ${100*max(ds1):+.0f}")
mac('SwapSigDriftSingleN', str(sum(x < 0 for x in ds1)))
bm = [S(b, d, 'bias5pct')['prem']['delta_mu_given_sigma_clean'][0] for b, d in SIX]; mac('SwapMuBias', rng([100*x for x in bm]))
bmu = [S(b, d, 'bias5pct')['unrec']['mu_perturbed_sigma_clean'][0] - S(b, d, 'bias5pct')['unrec']['C00_clean'][0] for b, d in SIX]; mac('SwapMuBiasUnrec', rng([100*x for x in bmu], 1))
bsu = [S(b, d, 'bias5pct')['unrec']['mu_clean_sigma_perturbed'][0] - S(b, d, 'bias5pct')['unrec']['C00_clean'][0] for b, d in SIX]; mac('SwapSigBiasUnrec', f"{100*min(bsu):+.1f}$ to ${100*max(bsu):+.1f}")
# ---------- rank stability by condition ---------------------------------------------------------------------------
byc = collections.defaultdict(lambda: [0, 0, 0])
for k, v in rk.items():
    c = k.split('|')[2]; byc[c][0] += v['reversal']; byc[c][1] += v['top_change']; byc[c][2] += 1
for c, nm in [('clean', 'Clean'), ('gaussian1pct', 'Noise'), ('drift5pct', 'Drift'), ('bias5pct', 'Bias')]:
    mac(f'Rev{nm}', str(byc[c][0])); mac(f'Top{nm}', str(byc[c][1])); mac(f'N{nm}', str(byc[c][2]))
mac('RevTotal', str(sum(v[0] for v in byc.values()))); mac('TopTotal', str(sum(v[1] for v in byc.values()))); mac('RevGroups', str(sum(v[2] for v in byc.values())))
pp = [v['p_boot_top5_ne_top100'] for v in rk.values() if v['top_change']]
mac('TopBootMed', f"{np.median(pp):.2f}"); mac('TopBootIQR', f"{np.percentile(pp,25):.2f}--{np.percentile(pp,75):.2f}"); mac('TopBootN', str(len(pp)))
mac('TopBootGeFive', str(sum(p >= 0.5 for p in pp)))
# ---------- label instant ------------------------------------------------------------------------------------------
mac('LabelMax', f"{100*lab['_max_abs_change']:.1f}")
hl = []
for b, d in SIX:
    for c in ['drift5pct', 'bias5pct']:
        v = lab[f'{b}|{d}|{c}|NLL|L20']; hl += [abs(v['prem_main'] - v['prem_cur']), abs(v['unrec_main'] - v['unrec_cur'])]
mac('LabelHeadline', f"{100*max(hl):.1f}")
# ---------- coverage vs f_oob --------------------------------------------------------------------------------------
pts = list(csv.DictReader(open(os.path.join(D, 'foob_coverage_points.csv'))))
def pt(bb, deg, ds, lv, m='NLL'):
    r = [x for x in pts if x['backbone'] == bb and x['degradation'] == deg and x['dataset'] == ds and x['mechanism'] == m and float(x['level_pct_range']) == lv][0]
    return float(r['f_oob']), float(r['picp'])
mi = J('foob_matched_interpolation.json', D)
def mi10(bb, ds, deg):
    v = mi[f'{bb}|{ds}|{deg}']['0.1']; return v
for deg in ['noise', 'bias', 'gain', 'drift']:
    vals = [mi10(bb, 'FD002', deg) for bb in BB]
    mac(f'Match{deg.capitalize()}', "--".join(f"{min(vals):.2f} {max(vals):.2f}".split()) if None not in vals else 'n/a')
    mac(f'Match{deg.capitalize()}L', f"{vals[0]:.2f}"); mac(f'Match{deg.capitalize()}T', f"{vals[1]:.2f}")
mac('MatchFDfourNoise', rng([mi10(bb, 'FD004', 'noise') for bb in BB], 2)); mac('MatchFDfourBias', rng([mi10(bb, 'FD004', 'bias') for bb in BB], 2)); mac('MatchFDfourGain', rng([mi10(bb, 'FD004', 'gain') for bb in BB], 2))
t2 = J('clean/table2_clean_full.json')
cpc = J('clean_protocol_comparison.json', D); clip = J('clip_sensitivity.json', D); cohsw = J('coherent_swap.json', D); coh = J('coherent_bias_control.json', D)
def clean_nll(bb, ds): return cpc[f'{bb}|{ds}']['B']['NLL']['picp']      # clean coverage under the stress-test processing (unclipped mean)
for ds, nm in [('FD001', 'One'), ('FD003', 'Three')]:
    vs = [pt(bb, 'drift', ds, 5)[1] for bb in BB]; mac(f'DriftCov{nm}', rng(vs, 2)); mac(f'CleanCov{nm}', rng([clean_nll(bb, ds) for bb in BB], 2))
mac('DriftFoobSingleMax', f"{max(pt(bb,'drift',ds,5)[0] for bb in BB for ds in ['FD001','FD003']):.1e}".replace('e-0', 'e-').replace('e-', '\\times10^{-')+'}')
g1 = [pt(bb, 'gain', ds, 0.1) for bb in BB for ds in ['FD001', 'FD003']]
mac('GainSingleFoob', rng([100*x[0] for x in g1])); mac('GainSinglePICP', rng([x[1] for x in g1], 2))
mac('DriftCovTwoSix', rng([pt(bb, 'drift', 'FD002', 5)[1] for bb in BB], 2)); mac('DriftCovFourSix', rng([pt(bb, 'drift', 'FD004', 5)[1] for bb in BB], 2))
mac('BiasCovSix', rng([pt(bb, 'bias', ds, 5)[1] for bb in BB for ds in ['FD002', 'FD004']], 2))
mac('BiasFoobSix', rng([100*pt(bb, 'bias', ds, 5)[0] for bb in BB for ds in ['FD002', 'FD004']])); mac('DriftFoobSix', rng([100*pt(bb, 'drift', ds, 5)[0] for bb in BB for ds in ['FD002', 'FD004']]))
mac('CleanCovSix', rng([clean_nll(bb, ds) for bb, ds in SIX], 2))
# left-censored crossover cells in the v1.8 Table V
hd = J('degradation/degradation_half_life_bias_gain_drift.json'); sw = J('degradation/degradation_sweep_bias_gain_drift.json')
def g(x):
    while isinstance(x, dict): x = x.get('grand_mean', list(x.values())[0])
    return float(x)
lc = []
for bb in BB:
    for deg in ['bias', 'drift', 'gain']:
        for ds in DS:
            if g(sw[bb][deg][ds]['NLL']['0.1']['picp']) < clean_nll(bb, ds) - 0.10: lc.append((bb, deg, ds))
mac('LCcells', str(len(lc))); mac('LCtotal', '24')
# ---------- engine-level CI half-widths ----------------------------------------------------------------------------------
# ---------- tables ---------------------------------------------------------------------------------------------------------
def f_ci(x, d=0): return f"{100*x[0]:.{d}f} [{100*x[1]:.{d}f}, {100*x[2]:.{d}f}]"
rows = []
for b, d in SIX:
    for c, cn in [('clean', 'Clean'), ('drift5pct', 'Drift 5\\%'), ('bias5pct', 'Bias 5\\%')]:
        r = B(b, d, c); pm = r['prem_by_model']
        ru = f"{r['rul_unc_at_trigger'][0]:.0f} [{r['rul_unc_at_trigger'][1]:.0f}, {r['rul_unc_at_trigger'][2]:.0f}]" if 'rul_unc_at_trigger' in r else '--'
        uc = f"{100*r['unrec_cond'][0]:.0f}" if r['unrec_cond'][0] is not None else '--'
        rows.append(f"{b} {d} & {cn} & {f_ci(r['prem'])} & {100*min(pm):.0f}--{100*max(pm):.0f} & {f_ci(r['unrec'],1)} & {uc} & {ru} \\\\")
    rows.append("\\midrule")
W('jrr_tab_decision', "\n".join(rows[:-1]) + "\n")
# Table: clean baseline summary (per backbone/mechanism over four sub-datasets)
fc = J('clean/fair_calibration_main_table.json')
def cl(bb, ds, m):
    r = cpc[f'{bb}|{ds}']['B'][m]; return r['picp'], r['mpiw'], r['is']      # same-path clean baseline (stress-test processing)
rows = []
for m, lab_ in METH:
    cells = []
    for bb in BB:
        v = [cl(bb, d, m) for d in DS]
        cells.append(f"{min(x[0] for x in v):.2f}--{max(x[0] for x in v):.2f} & {np.mean([x[1] for x in v]):.0f} & {np.mean([x[2] for x in v]):.1f}")
    rows.append(f"{lab_} & " + " & ".join(cells) + " \\\\")
W('jrr_tab_clean', "\n".join(rows) + "\n")
mac('EnsBestISeight', str(sum(1 for bb in BB for ds in DS if min((cl(bb, ds, m)[2], m) for m, _ in METH)[1] == 'Deep_Ensemble')))
# Supplement: corrected crossover table with left-censoring flags
rows = []
for bb in BB:
    for k, deg in enumerate(['bias', 'drift', 'gain']):
        cells = []
        for ds in DS:
            v = hd[bb][deg][ds]['NLL']['rel_co']; p01 = g(sw[bb][deg][ds]['NLL']['0.1']['picp']); fo01 = g(sw[bb][deg]['FD00' + ds[-1]]['feat_oob']['0.1'])
            if p01 < clean_nll(bb, ds) - 0.10: cells.append(f"LC ({100*fo01:.0f}\\%)")
            elif v is None: cells.append('$<10^{-4}$' if deg == 'drift' and g(sw[bb][deg][ds]['feat_oob']['5']) < 1e-4 else 'n.r.')
            else: cells.append(f"{100*v:.2f}\\%")
        rows.append((f"\\multirow{{3}}{{*}}{{{bb}}} & " if k == 0 else " & ") + deg.capitalize() + " & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
W('jrr_supp_types', "\n".join(rows[:-1]) + "\n")
# Supplement: matched-f_oob interpolation table (heteroscedastic head)
rows = []
for bb in BB:
    for ds in DS:
        for tgt in ('0.05', '0.1'):
            vals = []
            for deg in ['noise', 'bias', 'gain', 'drift']:
                v = mi[f'{bb}|{ds}|{deg}'][tgt]; vals.append('--' if v is None else f"{v:.2f}")
            rows.append(f"{bb} & {ds} & {int(float(tgt)*100)}\\% & " + " & ".join(vals) + " \\\\")
W('jrr_supp_matched', "\n".join(rows) + "\n")
# Supplement: output swap table (heteroscedastic head, L=20)
rows = []
for bb, ds in SIX + [(b, d) for b in BB for d in ['FD001', 'FD003']]:
    for c, cn in [('drift5pct', 'Drift'), ('bias5pct', 'Bias')]:
        s = S(bb, ds, c); p = s['prem']; u = s['unrec']
        f = lambda x: f"{100*x[0]:.1f}"
        fd = lambda x: f"{100*x[0]:+.1f} [{100*x[1][0]:+.1f}, {100*x[1][1]:+.1f}]"
        rows.append(f"{bb} {ds} & {cn} & {s['mean_mu_shift']:+.1f} & {f(p['C00_clean'])} & {f(p['mu_perturbed_sigma_clean'])} & {f(p['mu_clean_sigma_perturbed'])} & {f(p['both_perturbed'])} & {fd(p['delta_sigma_given_mu_pert'])} & {f(u['mu_perturbed_sigma_clean'])} & {f(u['mu_clean_sigma_perturbed'])} & {f(u['both_perturbed'])} \\\\")
W('jrr_supp_swap', "\n".join(rows) + "\n")
# Supplement: rank stability per condition
rows = []
for c, cn in [('clean', 'Clean'), ('gaussian1pct', 'Noise 1\\%'), ('drift5pct', 'Drift 5\\%'), ('bias5pct', 'Bias 5\\%')]:
    ks = [k for k in rk if k.split('|')[2] == c]
    pb = [rk[k]['p_boot_top5_ne_top100'] for k in ks if rk[k]['top_change']]
    rows.append(f"{cn} & {byc[c][2]} & {byc[c][0]} & {byc[c][1]} & " + (f"{np.median(pb):.2f}" if pb else '--') + " \\\\")
W('jrr_supp_rank', "\n".join(rows) + "\n")
# Supplement: full decision table with engine-bootstrap CIs (heteroscedastic head, all L, all conditions)
rows = []
for b in BB:
    for d in DS:
        for c, cn in [('clean', 'Clean'), ('gaussian1pct', 'Noise 1\\%'), ('drift5pct', 'Drift 5\\%'), ('bias5pct', 'Bias 5\\%')]:
            for L in (10, 20, 30):
                r = B(b, d, c, 'NLL', L)
                rows.append(f"{b} & {d} & {cn} & {L} & {f_ci(r['prem'],1)} & {f_ci(r['unrec'],1)}  \\\\")
W('jrr_supp_decision_nll', "\n".join(rows) + "\n")
# Supplement: all mechanisms at L=20 for drift and bias, six-condition
rows = []
for b, d in SIX:
    for c, cn in [('drift5pct', 'Drift'), ('bias5pct', 'Bias')]:
        for m, ml in METH:
            r = B(b, d, c, m)
            rows.append(f"{b} {d} & {cn} & {ml} & {f_ci(r['prem'])} & {f_ci(r['unrec'],1)} \\\\")
W('jrr_supp_mech', "\n".join(rows) + "\n")
# ---------- coherent-shift control (results_jrr/derived/coherent_bias_control.json) ------------------------------------
coh = J('coherent_bias_control.json', D)
def cv(bb, ds, v, k): return coh[f'{bb}|{ds}'][v][k]
SIXk = [f'{b}|{d}' for b, d in SIX]
mac('CohRampPICPSix', rng([coh[k]['ramp']['picp'] for k in SIXk], 2)); mac('CohPosPICPSix', rng([coh[k]['const_pos']['picp'] for k in SIXk], 2)); mac('CohNegPICPSix', rng([coh[k]['const_neg']['picp'] for k in SIXk], 2))
mac('CohRampPremSix', rng([100*coh[k]['ramp']['prem'][0] for k in SIXk])); mac('CohPosPremSix', rng([100*coh[k]['const_pos']['prem'][0] for k in SIXk])); mac('CohNegPremSixMax', f"{100*max(coh[k]['const_neg']['prem'][0] for k in SIXk):.0f}")
mac('CohNegUnrecSix', rng([100*coh[k]['const_neg']['unrec'][0] for k in SIXk])); mac('CohPosUnrecSixMax', f"{100*max(coh[k]['const_pos']['unrec'][0] for k in SIXk):.1f}")
SING = [f'{b}|{d}' for b in BB for d in ['FD001', 'FD003']]
mac('CohRampPICPSingle', rng([coh[k]['ramp']['picp'] for k in SING], 2)); mac('CohPosPICPSingle', rng([coh[k]['const_pos']['picp'] for k in SING], 2)); mac('CohNegPICPSingle', rng([coh[k]['const_neg']['picp'] for k in SING], 2))
mac('CohRampPICPThree', rng([coh[f'{b}|FD003']['ramp']['picp'] for b in BB], 2)); mac('CohPosPICPThree', rng([coh[f'{b}|FD003']['const_pos']['picp'] for b in BB], 2))
mac('CohRampPICPOne', rng([coh[f'{b}|FD001']['ramp']['picp'] for b in BB], 2)); mac('CohPosPICPOne', rng([coh[f'{b}|FD001']['const_pos']['picp'] for b in BB], 2))
mac('CohRampPremFDtwo', rng([100*coh[f'{b}|FD002']['ramp']['prem'][0] for b in BB])); mac('CohPosPremFDtwo', rng([100*coh[f'{b}|FD002']['const_pos']['prem'][0] for b in BB]))
mac('CohRampPremFDfour', rng([100*coh[f'{b}|FD004']['ramp']['prem'][0] for b in BB])); mac('CohPosPremFDfour', rng([100*coh[f'{b}|FD004']['const_pos']['prem'][0] for b in BB]))
mac('CohRampPICPFDtwo', rng([coh[f'{b}|FD002']['ramp']['picp'] for b in BB], 2)); mac('CohPosPICPFDtwo', rng([coh[f'{b}|FD002']['const_pos']['picp'] for b in BB], 2))
mac('CohNegUnrecFDtwo', rng([100*coh[f'{b}|FD002']['const_neg']['unrec'][0] for b in BB])); mac('CohNegUnrecFDfour', rng([100*coh[f'{b}|FD004']['const_neg']['unrec'][0] for b in BB]))
rows = []
for b in BB:
    for d in DS:
        c = coh[f'{b}|{d}']; f = lambda v, k, dd=2: f"{c[v][k]:.{dd}f}"
        pr = lambda v: f"{100*c[v]['prem'][0]:.0f}"; ur = lambda v: f"{100*c[v]['unrec'][0]:.1f}"
        rows.append(f"{b} {d} & {f('clean','picp')} & {f('ramp','picp')} / {pr('ramp')} / {ur('ramp')} & {f('const_pos','picp')} / {pr('const_pos')} / {ur('const_pos')} & {f('const_neg','picp')} / {pr('const_neg')} / {ur('const_neg')} \\\\")
    rows.append("\\midrule")
W('jrr_tab_coh', "\n".join(rows[:-1]) + "\n")
mac('CleanCovAll', rng([cl(bb, ds, m)[0] for bb in BB for ds in DS for m, _ in METH], 2))
# ---------- fair-calibration table regenerated from JSON (v1.8 hard-typed this table in main.tex) ----------------------
fsig = J('clean/fair_calibration_sigma_fixed.json'); trr = J('clean/lstm_mse_fixed_training_residual_per_seed.json')
rows = []
for bb in BB:
    for ds in DS:
        ratio = fsig[bb][ds]['mean_ratio_calib_over_train']
        trp = f"{trr[ds]['picp_mean']:.3f}" if bb == 'LSTM' and ds in trr else '--'
        rows.append(f"{bb} {ds} & {ratio:.3f} & {trp} & {fc[bb][ds]['MSE_fixed']['picp_mean']:.3f} & {fc[bb][ds]['MC_Dropout_fixed']['picp_mean']:.3f} \\\\")
W('jrr_supp_fair', "\n".join(rows) + "\n")

# ---------- P1: predicted-mean processing (clean same-path baseline, clipping sensitivity) -------------------------------
KEYS = [f'{bb}|{ds}' for bb in BB for ds in DS]
# (i) full clean table under the stress-test processing, one row per backbone x sub-dataset x mechanism
rows = []; csvrows = [['backbone', 'dataset', 'mechanism', 'picp_stress_processing', 'mpiw', 'ece', 'is', 'rmse', 'picp_v18_benchmark_processing', 'is_v18_table']]
for bb in BB:
    for ds in DS:
        for m, lab_ in METH:
            b = cpc[f'{bb}|{ds}']['B'][m]; a = cpc[f'{bb}|{ds}']['A'][m]
            rows.append(f"{bb} {ds} & {lab_} & {b['picp']:.3f} & {b['mpiw']:.1f} & {b['ece']:.3f} & {b['is']:.1f} & {b['rmse']:.2f} & {t2[bb][ds][m]['picp']:.3f} \\\\")
            csvrows.append([bb, ds, m, b['picp'], b['mpiw'], b['ece'], b['is'], b['rmse'], t2[bb][ds][m]['picp'], t2[bb][ds][m]['is']])
        rows.append("\\midrule") if not (bb == BB[-1] and ds == DS[-1]) else None
W('jrr_supp_clean_same', "\n".join(rows) + "\n")
with open(os.path.join(D, 'clean_same_path_table.csv'), 'w', newline='') as fh: csv.writer(fh, lineterminator='\n').writerows(csvrows)
dl = [cpc[k]['B']['NLL']['picp'] - cpc[k]['A']['NLL']['picp'] for k in KEYS]
mac('CleanNLLShiftMax', f"{-100*min(dl):.1f}"); mac('CleanNLLShiftNchg', str(sum(abs(x) > 1e-9 for x in dl)))
allsh = [cpc[k]['B'][m]['picp'] - cpc[k]['A'][m]['picp'] for k in KEYS for m, _ in METH]
mac('CleanAllShiftMax', f"{-100*min(allsh):.1f}"); mac('CleanAllShiftNchg', str(sum(abs(x) > 1e-9 for x in allsh))); mac('CleanAllShiftN', str(len(allsh)))
mac('CleanRMSEshiftMax', f"{max(cpc[k]['B'][m]['rmse'] - cpc[k]['A'][m]['rmse'] for k in KEYS for m, _ in METH):.2f}")
# agreement of the clean NLL coverage across three code paths: same-path benchmark (B), coherent-control clean row, released drift controls
ctl = J('degradation/drift_controls.json')
xc = [abs(coh[k]['clean']['picp'] - cpc[k]['B']['NLL']['picp']) for k in KEYS]
mac('CleanXcheckCoh', f"{max(xc):.4f}")
xr = [abs(ctl[k.split('|')[0]][k.split('|')[1]]['clean_picp_NLL'] - cpc[k]['B']['NLL']['picp']) for k in KEYS if k.split('|')[1] in ('FD001', 'FD002')]
mac('CleanXcheckCtl', f"{max(xr):.4f}")
# (ii) clipping sensitivity from the stored per-engine outputs
sm = clip['summary']
mac('ClipRateMaxNLL', f"{100*sm['max_abs_rate_diff_by_mechanism']['NLL']:.1f}"); mac('ClipRateMaxCP', f"{100*sm['max_abs_rate_diff_by_mechanism']['CP_norm']:.1f}")
mac('ClipRateMaxEns', f"{100*sm['max_abs_rate_diff_by_mechanism']['Deep_Ensemble']:.1f}"); mac('ClipRateN', str(sm['n_cells']))
mac('ClipPICPmaxClean', f"{sm['picp_abs_diff_max_by_cond']['clean']:.3f}"); mac('ClipPICPmaxNoise', f"{sm['picp_abs_diff_max_by_cond']['gaussian1pct']:.3f}")
mac('ClipPICPmaxDrift', f"{sm['picp_abs_diff_max_by_cond']['drift5pct']:.3f}"); mac('ClipPICPmaxBias', f"{sm['picp_abs_diff_max_by_cond']['bias5pct']:.3f}")
dr2 = [clip['picp'][f'{bb}|FD002|drift5pct'] for bb in BB]
mac('ClipDriftTwoU', rng([x['U'] for x in dr2], 2)); mac('ClipDriftTwoA', rng([x['A'] for x in dr2], 2))
rows = []
CONDS_C = ['clean', 'gaussian1pct', 'bias5pct', 'drift5pct']
for k in KEYS:
    cells = " & ".join(f"{clip['picp'][f'{k}|{c}']['U']:.3f} / {clip['picp'][f'{k}|{c}']['A']:.3f}" for c in CONDS_C)
    rows.append(f"{k.replace('|', ' ')} & {cells} \\\\")
W('jrr_supp_clip', "\n".join(rows) + "\n")
# ---------- P1: waveform controls on FD001 and FD002 from the released controls -----------------------------------------
fx = J('degradation/drift_fixed_endpoint_shuffle_control.json')
rows = []
for bb in BB:
    for ds in ['FD001', 'FD002']:
        c = ctl[bb][ds]; ramp = g(sw[bb]['drift'][ds]['NLL']['5']['picp']); cp_ = coh[f'{bb}|{ds}']['const_pos']['picp']
        fe = " & ".join(f"{fx[bb][k]:.3f}" for k in ['1', '3', '5']) if ds == 'FD002' else "-- & -- & --"
        rows.append(f"{bb} {ds} & {c['clean_picp_NLL']:.3f} & {ramp:.3f} & {cp_:.3f} & {c['reverse']['picp_NLL']:.3f} & {c['shuffled']['picp_NLL']:.3f} & {c['continuous']['picp_NLL']:.3f} & {c['bias_equal']['picp_NLL']:.3f} & {c['singlech']['picp_NLL']:.3f} & {fe} \\\\")
W('jrr_supp_waveform', "\n".join(rows) + "\n")
def ctlv(key, ds): return [ctl[bb][ds][key]['picp_NLL'] for bb in BB]
mac('WFoneRamp', rng([g(sw[bb]['drift']['FD001']['NLL']['5']['picp']) for bb in BB], 2)); mac('WFoneShuf', rng(ctlv('shuffled', 'FD001'), 2)); mac('WFoneRev', rng(ctlv('reverse', 'FD001'), 2))
mac('WFoneCont', rng(ctlv('continuous', 'FD001'), 2)); mac('WFoneSingle', rng(ctlv('singlech', 'FD001'), 2)); mac('WFoneBiasEq', rng(ctlv('bias_equal', 'FD001'), 2))
mac('WFtwoShufL', f"{ctl['LSTM']['FD002']['shuffled']['picp_NLL']:.2f}"); mac('WFtwoShufT', f"{ctl['Transformer']['FD002']['shuffled']['picp_NLL']:.2f}")
mac('WFoneClean', rng([ctl[bb]['FD001']['clean_picp_NLL'] for bb in BB], 2))
# ---------- P2: scale effect extremes and sign-pattern output swap ----------------------------------------------------------
def sw_sig(b, d): v = S(b, d, 'drift5pct')['prem']['delta_sigma_given_mu_pert']; return 100*v[0], 100*v[1][0], 100*v[1][1]
ext = {(b, d): sw_sig(b, d) for b in BB for d in DS}
kmax = max(ext, key=lambda k: ext[k][0]); kmin = min(ext, key=lambda k: ext[k][0])
def sgn(x, d=1): return f"${x:+.{d}f}$"
mac('SigExtMaxCell', f"{kmax[0]} {kmax[1]}"); mac('SigExtMaxPt', sgn(ext[kmax][0])); mac('SigExtMaxLo', sgn(ext[kmax][1])); mac('SigExtMaxHi', sgn(ext[kmax][2]))
mac('SigExtMinCell', f"{kmin[0]} {kmin[1]}"); mac('SigExtMinPt', sgn(ext[kmin][0])); mac('SigExtMinLo', sgn(ext[kmin][1])); mac('SigExtMinHi', sgn(ext[kmin][2]))
nsig = [(b, d) for (b, d) in ext if ext[(b, d)][1] > 0 or ext[(b, d)][2] < 0]; mac('SigExtExclZero', str(len(nsig))); mac('SigExtCells', str(len(ext)))
six_neg = [cohsw[f'{b}|{d}']['const_neg'] for b, d in SIX]
mac('CNswapMuUnrec', rng([100*r['unrec']['delta_mu_given_sigma_clean'][0] for r in six_neg], 0))
mac('CNswapSigUnrec', sgn(100*min(r['unrec']['delta_sigma_given_mu_pert'][0] for r in six_neg)) + ' to ' + sgn(100*max(r['unrec']['delta_sigma_given_mu_pert'][0] for r in six_neg)))
mac('CNswapShift', f"{min(r['mean_shift'] for r in six_neg):.0f}--{max(r['mean_shift'] for r in six_neg):.0f}")
six_pos = [cohsw[f'{b}|{d}']['const_pos'] for b, d in SIX]
mac('CPswapMuPrem', rng([100*r['prem']['delta_mu_given_sigma_clean'][0] for r in six_pos], 0)); mac('CPswapShift', f"{-max(r['mean_shift'] for r in six_pos):.0f}--{-min(r['mean_shift'] for r in six_pos):.0f}")   # magnitude of the (negative) mean shift
mac('CPswapSigPrem', sgn(100*min(r['prem']['delta_sigma_given_mu_pert'][0] for r in six_pos), 0) + ' to ' + sgn(100*max(r['prem']['delta_sigma_given_mu_pert'][0] for r in six_pos), 0))
rows = []
for bb in BB:
    for ds in DS:
        for vn, vl in [('ramp', 'Ramp (+)'), ('const_pos', 'Const.\\ $+$'), ('const_neg', 'Const.\\ $-$')]:
            r = cohsw[f'{bb}|{ds}'][vn]; f3 = lambda x: f"{100*x[0]:+.1f} [{100*x[1]:+.1f}, {100*x[2]:+.1f}]"
            rows.append(f"{bb} {ds} & {vl} & {r['mean_shift']:+.1f} & {r['sigma_ratio']:.2f} & {f3(r['prem']['delta_mu_given_sigma_clean'])} & {f3(r['prem']['delta_sigma_given_mu_pert'])} & {f3(r['unrec']['delta_mu_given_sigma_clean'])} & {f3(r['unrec']['delta_sigma_given_mu_pert'])} \\\\")
    rows.append("\\midrule")
W('jrr_supp_cohswap', "\n".join(rows[:-1]) + "\n")
# ---------- P2: all 96 rank-stability probabilities (machine-readable) -------------------------------------------------------
with open(os.path.join(D, 'rank_stability_all96.csv'), 'w', newline='') as fh:
    w = csv.writer(fh, lineterminator='\n'); w.writerow(['backbone', 'dataset', 'condition', 'L', 'top_changes_between_ratios', 'ranking_reverses', 'p_boot_top_ratio5_ne_top_ratio100', 'top_ratio5', 'top_ratio20', 'top_ratio100'])
    for k, v in rk.items():
        bb_, ds_, c_, L_ = k.split('|'); w.writerow([bb_, ds_, c_, L_, v['top_change'], v['reversal'], v['p_boot_top5_ne_top100'], v['top_by_ratio']['5'], v['top_by_ratio']['20'], v['top_by_ratio']['100']])
mac('RankCSVn', str(len(rk)))

with open(os.path.join(D, 'decision_engine_bootstrap_all.csv'), 'w', newline='') as fh:
    w = csv.writer(fh, lineterminator='\n'); w.writerow(['backbone', 'dataset', 'condition', 'mechanism', 'L', 'premature', 'premature_lo', 'premature_hi', 'unrecognised', 'unrecognised_lo', 'unrecognised_hi'])
    for k, v in boot.items():
        bb_, ds_, c_, m_, L_ = k.split('|'); w.writerow([bb_, ds_, c_, m_, L_.lstrip('L'), *v['prem'], *v['unrec']])
mac('BootCSVn', str(len(boot)))
with open(os.path.join(T, 'numbers_jrr.tex'), 'w') as f:
    for k, v in M.items(): f.write(f"\\newcommand{{\\J{k}}}{{{v}}}\n")
for k, v in M.items(): print(f"{k:22s} {v}")
