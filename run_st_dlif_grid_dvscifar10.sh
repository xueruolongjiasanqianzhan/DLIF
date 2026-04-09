#!/usr/bin/env bash
set -euo pipefail

# ==========================
# ST-DLIF grid search runner
# ==========================
# Usage:
#   bash run_st_dlif_grid_dvscifar10.sh
#
# Optional overrides:
#   DATA_DIR=/path/to/CIFAR10DVS BASE_OUT=./logs_grid bash run_st_dlif_grid_dvscifar10.sh

DATA_DIR="${DATA_DIR:-/home/guyue/zhangenle/CIFAR10DVS}"
BASE_OUT="${BASE_OUT:-./logs_grid_dvs_st}"
SCRIPT_PATH="DVS-CIFAR10/SLTT.py"

COMMON_ARGS=(
  # Paper-aligned fixed training setup (non-grid dimensions)
  -seed 1
  -dataset DVSCIFAR10
  -data_dir "$DATA_DIR"
  -model spiking_vgg11_bn
  -surrogate triangle
  -T 10
  -tau 1.1
  -b 32
  -epochs 200
  -j 4
  -amp
  -opt SGD
  -lr 0.1
  -momentum 0.9
  -lr_scheduler CosALR
  -T_max 200
  -weight_decay 0.0
  -drop_rate 0.0
  -loss_lambda 0.05
  -st_dlif_enabled
  -st_dlif_activation tanh
)

# Small grid (16 runs): mode x gamma x sparsity x detach
modes=("additive" "event")
gammas=("0.05" "0.1")
sparsities=("0.8" "0.9")
detaches=("detach" "no_detach")

mkdir -p "$BASE_OUT"

for mode in "${modes[@]}"; do
  for gamma in "${gammas[@]}"; do
    for sp in "${sparsities[@]}"; do
      for det in "${detaches[@]}"; do
        run_name="m-${mode}_g-${gamma}_sp-${sp}_det-${det}"
        out_dir="${BASE_OUT}/${run_name}"

        cmd=(python "$SCRIPT_PATH"
          "${COMMON_ARGS[@]}"
          -st_dlif_mode "$mode"
          -st_dlif_gamma_init "$gamma"
          -bilinear_sparsity_level "$sp"
          -out_dir "$out_dir"
          -name "$run_name"
        )

        if [[ "$det" == "no_detach" ]]; then
          cmd+=(--no_st_dlif_detach_prev)
        fi

        echo "=================================================="
        echo "Running: $run_name"
        echo "Command: ${cmd[*]}"
        "${cmd[@]}"
      done
    done
  done
done

echo "All grid runs finished."
