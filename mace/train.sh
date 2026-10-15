#!/bin/bash
# Train the small Al MACE model exactly as the published one (mace/models/Al_small_stagetwo.model):
# mace-torch 0.3.16, seed 1, CPU, float64, 16 threads, 64 channels (64x0e + 64x1o), on
# data/Al/subset_20_percent.extxyz (1,579 frames; MACE holds 5% out for validation with the same seed).
# Every argument not given here is a mace-torch 0.3.16 default; the published run's full resolved
# configuration is mace/models/train_config.txt and its log mace/models/train_log.txt.  ~5.5 h.
#
#   OUT=<route root> bash mace/train.sh        (run.sh refit submits this to SLURM)
#
# Output: $OUT/mace/models/Al_small_stagetwo.model, the final stage-two (SWA) model every later step uses,
# and $OUT/mace/train/ (MACE's logs, per-step results, checkpoints).  Bit-identical weights are not
# guaranteed across hardware or thread counts (floating-point reduction order): compare the error tables
# in the log with mace/models/train_log.txt, and the downstream results statistically (run.sh verify).
set -euo pipefail
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:-$(dirname "$D")/repro/saved}
PY=${PY:-$(dirname "$D")/.venv/bin/python}
DATA=$(dirname "$D")/data/Al/subset_20_percent.extxyz
[ -f "$DATA" ] || { echo "missing $DATA — run: bash run.sh data"; exit 1; }
M=$OUT/mace/models; T=$OUT/mace/train
mkdir -p "$M" "$T"
cd "$T" && OMP_NUM_THREADS=16 "$PY" -m mace.cli.run_train \
  --name Al_small --train_file "$DATA" --valid_fraction 0.05 --seed 1 \
  --energy_key dft_energy --forces_key dft_forces --E0s average \
  --model MACE --hidden_irreps '64x0e + 64x1o' --r_max 6.0 --num_interactions 2 --max_L 1 \
  --correlation 3 --MLP_irreps 16x0e \
  --batch_size 16 --max_num_epochs 400 --patience 50 --eval_interval 5 --ema --swa \
  --default_dtype float64 --device cpu --save_cpu \
  --model_dir "$M" --checkpoints_dir "$T/checkpoints" --results_dir "$T/results" --log_dir "$T/logs"
