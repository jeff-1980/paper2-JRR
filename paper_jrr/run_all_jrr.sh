#!/bin/bash
# Full JRR build from repo root.  Steps 1-2 need C-MAPSS (RUL_DATA_DIR), PyTorch and the released checkpoints; they only
# run inference (no training).  Steps 3-6 need only released result files + results_jrr/derived/.
set -e
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
if [ -n "$RUL_DATA_DIR" ]; then
  python code_jrr/jrr_decision_per_engine.py          # 1. per-engine lower bounds (all 8 backbone x sub-dataset pickles)
  python code_jrr/jrr_coherent_bias_control.py        # 2. sign-coherent shift control
  python code_jrr/jrr_coherent_swap.py                # 2b. output swap for the sign-pattern variants
  python code_jrr/jrr_clean_same_path.py              # 2c. clean baseline under the stress-test mean processing (+ v1.8 clipping variant)
fi
python code_jrr/jrr_clip_sensitivity.py               # 2d. mean-clipping sensitivity from the stored per-engine outputs (no GPU)
python code_jrr/jrr_derived_stats.py                  # 3. bootstrap, output swap, label-instant, rank stability, f_oob points
python paper_jrr/make_tables_v18.py .                 # 4a. unchanged v1.8 table generator -> paper_jrr/tables/
python paper_jrr/make_tables_jrr.py .                 # 4b. new / re-derived tables and J-macros
python paper_jrr/make_figures_jrr.py .                # 5. figures 1-3
python paper_jrr/count_words_jrr.py                   # 6a. word counts
bash paper_jrr/build_jrr.sh                           # 6b. PDFs (main named/anonymous, title page, supplement named/anonymous)
